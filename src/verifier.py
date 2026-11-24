"""Stage 2: the verifier (a cross-encoder), and the two-stage pipeline.

The verifier reads a sentence and one candidate technique together:
  <s> sentence </s></s> technique text </s>
A linear head on the first token gives a logit; its sigmoid is the support probability
p(s, t). Trained with binary cross-entropy on the pairs from src/mine_negatives.py.
The sentence is cut to 128 tokens and the technique text to 256 (every in-scope text
fits), so the description always stays whole.

Pipeline (the `predict` command): take the retriever's top-k candidates per sentence,
keep a candidate when p(s, t) >= theta_v. k and theta_v are tuned on dev only.

Usage:
  python -m src.verifier train   --run full --seed 0
  python -m src.verifier predict --run full --seed 0
Add --smoke for a quick check that the code runs (writes to runs/smoke/verifier/).
Uses runs/<run>/retriever/seed<N>/candidates_*.jsonl and runs/<run>/verifier/seed<N>/pairs_train.jsonl.

Outputs in runs/<run>/verifier/seed<N>/:
  model.pt, config.json, history.json      training (history: per epoch, train loss; dev loss on
                                           dev pairs mined like the training pairs; dev pipeline
                                           scores; pipeline scores on a fixed train sample, with
                                           thresholds tuned on that sample; for evaluation/curves.py)
  thresholds.json                          k and theta_v picked on dev
  dev.jsonl, test.jsonl                    two-stage predictions for evaluation/evaluate.py
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from evaluation.evaluate import write_predictions
from src.mine_negatives import mine
from src.common import (add_smoke_option, load_jsonl, load_split, model_dir, pick_device, save_json, set_seed, technique_pool,
                        technique_texts, tune_thresholds)

BASE = "ehsanaghaei/SecureBERT"
SENT_LEN = 128
TECH_LEN = 256
KS = (3, 5, 10)
PAD_TO = 32


def free_gpu_memory(device):
    if device.type == "mps":
        torch.mps.empty_cache()


def gpu_memory(device):
    """GPU memory in use, for the progress line (Mac GPU only)."""
    return f"  gpu {torch.mps.driver_allocated_memory() / 1e9:.1f} GB" if device.type == "mps" else ""


class Verifier(nn.Module):
    def __init__(self, base=BASE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        first = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(self.dropout(first)).squeeze(-1)


class PairBuilder:
    """Turns (sentence text, technique id) into model input. Each text is tokenized once."""

    def __init__(self, tokenizer, techniques):
        self.tokenizer = tokenizer
        self.tech = {t: self._ids(text, TECH_LEN) for t, text in technique_texts(techniques).items()}
        self.sentences = {}

    def _ids(self, text, max_len):
        return self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]

    def ids(self, sentence, technique):
        if sentence not in self.sentences:
            self.sentences[sentence] = self._ids(sentence, SENT_LEN)
        # RoBERTa pair layout (SecureBERT is a RoBERTa model): <s> A </s></s> B </s>
        cls, sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        return [cls, *self.sentences[sentence], sep, sep, *self.tech[technique], sep]

    def batch(self, pairs, device):
        seqs = [self.ids(s, t) for s, t in pairs]
        # Round the width up to a multiple of PAD_TO: only a few distinct shapes reach the GPU,
        # so the Mac's GPU reuses memory instead of reserving new memory for every length.
        width = -(-max(map(len, seqs)) // PAD_TO) * PAD_TO
        input_ids = torch.full((len(seqs), width), self.tokenizer.pad_token_id)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return input_ids.to(device), mask.to(device)


@torch.no_grad()
def score_pairs(model, builder, pairs, device, batch_size=64):
    """Support probability for each (sentence text, technique id) pair."""
    model.eval()
    # Sort by length so each batch pads little; put results back in the original order.
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
    probs = np.zeros(len(pairs))
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        logits = model(*builder.batch([pairs[i] for i in idx], device))
        probs[idx] = torch.sigmoid(logits).float().cpu().numpy()
    return probs


def load_candidates(retriever_dir, split):
    """{(doc_id, sent_idx): [(technique, retriever score), ...]} in retriever rank order."""
    path = retriever_dir / f"candidates_{split}.jsonl"
    return {(c["doc_id"], c["sent_idx"]): [tuple(x) for x in c["candidates"]] for c in load_jsonl(path)}


def verify(model, builder, rows, candidates, device):
    """{key: [(technique, p), ...]} in retriever rank order, for every row."""
    pairs = [(r["text"], t) for r in rows for t, _ in candidates[(r["doc_id"], r["sent_idx"])]]
    probs = iter(score_pairs(model, builder, pairs, device))
    return {(r["doc_id"], r["sent_idx"]): [(t, float(next(probs))) for t, _ in candidates[(r["doc_id"], r["sent_idx"])]]
            for r in rows}


def pipeline_matrix(rows, verified, pool, k):
    """S: verifier probability for the retriever's top-k candidates, -inf elsewhere. G: gold."""
    col = {t: j for j, t in enumerate(pool)}
    S = np.full((len(rows), len(pool)), -np.inf)
    G = np.zeros((len(rows), len(pool)), dtype=bool)
    for i, r in enumerate(rows):
        for t, p in verified[(r["doc_id"], r["sent_idx"])][:k]:
            S[i, col[t]] = p
        for t in r["labels"]:
            G[i, col[t]] = True
    return S, G


def tune_pipeline(rows, verified, pool, metric):
    """Best k (retriever candidates kept) and theta_v on these rows. Dev only."""
    best = None
    for k in KS:
        S, G = pipeline_matrix(rows, verified, pool, k)
        result = tune_thresholds(S, G, metric=metric)
        result["k"] = k
        if best is None or result[metric] > best[metric]:
            best = result
    return best


def mined_pairs(rows, candidates, mining):
    """(text, technique, label) pairs mined from `rows` with the training pairs' settings (mining.json)."""
    keyed = {(r["doc_id"], r["sent_idx"]): r for r in rows}
    pairs, _, _ = mine(keyed, {k: candidates[k] for k in keyed}, mining.get("neg_k", 5), mining.get("unlabeled_k", 3),
                       mining.get("unlabeled_frac", 0.4), mining.get("guard_quantile", 0.75), mining.get("seed", 0))
    return [(keyed[p["key"]]["text"], p["technique"], float(p["label"])) for p in pairs]


def pair_loss(model, builder, pairs, device):
    """Mean binary cross-entropy over (text, technique, label) pairs, in eval mode. Same loss as training."""
    probs = np.clip(score_pairs(model, builder, [(s, t) for s, t, _ in pairs], device), 1e-7, 1 - 1e-7)
    y = np.array([label for *_, label in pairs])
    return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def train_candidates_path(out, retriever_dir):
    """The train candidates the training pairs were mined from (recorded in mining.json)."""
    mining_file = out / "mining.json"
    if mining_file.exists() and "candidates" in (m := json.loads(mining_file.read_text())):
        return Path(m["candidates"])
    return retriever_dir / "candidates_train_crossfit.jsonl"


def load_model(path, device):
    model = Verifier()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


# ---------------------------------------------------------------- train

def train(args):
    set_seed(args.seed)
    device = pick_device()
    out = model_dir(args, "verifier")
    pool = technique_pool(args.run, "train")

    text = {(r["doc_id"], r["sent_idx"]): r["text"] for r in load_split("train", args.run)}
    pairs = load_jsonl(out / "pairs_train.jsonl")
    if args.limit:
        pairs = pairs[:args.limit]
    examples = [(text[(p["doc_id"], p["sent_idx"])], p["technique"], float(p["label"])) for p in pairs]
    assert all(t in pool for _, t, _ in examples), "training pair outside the training pool"

    dev = load_split("dev", args.run, args.dev_limit)
    dev_pool = technique_pool(args.run, "dev")
    dev_candidates = load_candidates(model_dir(args, "retriever"), "dev")
    mining = json.loads((out / "mining.json").read_text()) if (out / "mining.json").exists() else {}
    dev_pairs = mined_pairs(dev, dev_candidates, mining)

    # A fixed sample of training sentences, run through the pipeline each epoch: the train side
    # of the train/dev gap. Thresholds are re-tuned on the sample itself (a diagnostic, not a result).
    train_candidates, train_sample = {}, []
    if args.train_sample:
        path = train_candidates_path(out, model_dir(args, "retriever"))
        train_candidates = {(c["doc_id"], c["sent_idx"]): [tuple(x) for x in c["candidates"]] for c in load_jsonl(path)}
        train_rows = [r for r in load_split("train", args.run) if (r["doc_id"], r["sent_idx"]) in train_candidates]
        sample_idx = np.random.default_rng(args.seed + 1).permutation(len(train_rows))[:args.train_sample]
        train_sample = [train_rows[i] for i in sorted(sample_idx)]
    positives = sum(y for *_, y in examples)
    print(f"{args.run} run | {len(examples)} pairs ({positives:.0f} positive) | {len(dev)} dev sentences | {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    builder = PairBuilder(tokenizer, sorted(set(pool) | set(dev_pool)))
    model = Verifier().to(device)
    steps_per_epoch = math.ceil(len(examples) / args.batch_size)
    total = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    best, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, losses = time.time(), []
        order = rng.permutation(len(examples))
        for b in range(steps_per_epoch):
            batch = [examples[i] for i in order[b * args.batch_size:(b + 1) * args.batch_size]]
            # Gradient accumulation: run the batch in pieces of --micro-batch pairs and add up the
            # gradients. Same update as one batch of --batch-size, with less GPU memory at a time.
            optimizer.zero_grad()
            batch_loss = 0.0
            for m in range(0, len(batch), args.micro_batch):
                piece = batch[m:m + args.micro_batch]
                logits = model(*builder.batch([(s, t) for s, t, _ in piece], device))
                loss = loss_fn(logits, torch.tensor([y for *_, y in piece], device=device)) * len(piece) / len(batch)
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            scheduler.step()
            losses.append(batch_loss)
            if (b + 1) % 100 == 0:
                free_gpu_memory(device)
            if args.log_every and (b + 1) % args.log_every == 0:
                print(f"  step {b + 1}/{steps_per_epoch}  loss {np.mean(losses[-args.log_every:]):.4f}"
                      f"  ({time.time() - start:.0f}s){gpu_memory(device)}", flush=True)

        free_gpu_memory(device)
        dev_best = tune_pipeline(dev, verify(model, builder, dev, dev_candidates, device), dev_pool, args.metric)
        dev_loss = pair_loss(model, builder, dev_pairs, device)
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "dev_loss": dev_loss, **dev_best}
        if train_sample:
            train_best = tune_pipeline(train_sample, verify(model, builder, train_sample, train_candidates, device),
                                       pool, args.metric)
            entry.update({f"train_{k}": v for k, v in train_best.items()})
        free_gpu_memory(device)
        entry["seconds"] = round(time.time() - start)
        history.append(entry)
        print(f"epoch {epoch}  loss {np.mean(losses):.4f}  dev loss {dev_loss:.4f}  dev {args.metric} {dev_best[args.metric]:.3f}"
              f" (k={dev_best['k']}, theta={dev_best['theta']:.2f}, P {dev_best['micro_precision']:.3f}"
              f" R {dev_best['micro_recall']:.3f})" +
              (f"  train {args.metric} {entry[f'train_{args.metric}']:.3f}" if train_sample else "") +
              f"  ({time.time() - start:.0f}s)", flush=True)
        if dev_best[args.metric] > best:
            best, bad_epochs = dev_best[args.metric], 0
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"no dev improvement for {args.patience} epochs, stopping")
                break

    save_json(out / "history.json", history)
    save_json(out / "config.json", {"model": "verifier", "base": BASE, "run": args.run, "seed": args.seed,
                                    "retriever": str(model_dir(args, "retriever")),
                                    "lr": args.lr, "batch_size": args.batch_size, "micro_batch": args.micro_batch, "epochs": args.epochs,
                                    "patience": args.patience, "selected_on": f"dev {args.metric}",
                                    "best_dev": best, "limit": args.limit, "pool": pool})
    print(f"best dev {args.metric} {best:.3f} -> {out / 'model.pt'}")


# ---------------------------------------------------------------- predict (the two-stage pipeline)

def predict(args):
    device = pick_device()
    out = model_dir(args, "verifier")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = load_model(out / "model.pt", device)
    if args.limit:
        print(f"--limit {args.limit}: predictions cover part of each split, evaluate.py will reject them")

    verified = {}
    for split in ("dev", "test"):
        rows = load_split(split, "full", args.limit)  # every sentence, as evaluate.py expects
        builder = PairBuilder(tokenizer, technique_pool(args.run, split))
        verified[split] = verify(model, builder, rows, load_candidates(model_dir(args, "retriever"), split), device)

    dev = [r for r in load_split("dev", args.run) if (r["doc_id"], r["sent_idx"]) in verified["dev"]]
    best = tune_pipeline(dev, verified["dev"], technique_pool(args.run, "dev"), args.metric)
    save_json(out / "thresholds.json", {"tuned_on": "dev", "metric": args.metric, **best})
    print(f"dev: k={best['k']} theta_v={best['theta']:.3f}  micro F1 {best['micro_f1']:.3f}  macro F1 {best['macro_f1']:.3f}"
          f"  (P {best['micro_precision']:.3f} R {best['micro_recall']:.3f})")

    for split in ("dev", "test"):
        write_predictions(out / f"{split}.jsonl", [
            {"doc_id": key[0], "sent_idx": key[1],
             "predicted": [t for t, p in cands[:best["k"]] if p >= best["theta"]],
             "scores": dict(cands)}
            for key, cands in verified[split].items()])
    print(f"predictions -> {out}/dev.jsonl, {out}/test.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("train", "predict"))
    ap.add_argument("--run", default="full", choices=("full", "withheld"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="train: first N training pairs; predict: first N sentences per split (smoke runs)")
    ap.add_argument("--dev-limit", type=int, help="use only the first N dev sentences during training")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=16,
                    help="pairs sent to the GPU at once; the batch is split into pieces this size (saves memory)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=2, help="epochs without dev improvement before stopping")
    ap.add_argument("--metric", default="micro_f1", choices=("micro_f1", "macro_f1"),
                    help="dev score for early stopping and for picking k and theta_v")
    ap.add_argument("--log-every", type=int, default=200, help="print the training loss every N steps (0: off)")
    ap.add_argument("--train-sample", type=int, default=1000,
                    help="train sentences run through the pipeline each epoch for the train/dev gap (0: off)")
    add_smoke_option(ap)
    args = ap.parse_args()
    if args.smoke:
        args.run, args.limit, args.dev_limit, args.epochs, args.log_every = "full", args.limit or 300, 300, 1, 0
        args.train_sample = min(args.train_sample, 200)
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
