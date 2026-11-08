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
Uses runs/<run>/retriever/seed<N>/candidates_*.jsonl and runs/<run>/verifier/seed<N>/pairs_train.jsonl.

Outputs in runs/<run>/verifier/seed<N>/:
  model.pt, config.json, history.json      training
  thresholds.json                          k and theta_v picked on dev
  dev.jsonl, test.jsonl                    two-stage predictions for evaluation/evaluate.py
"""
import argparse
import math
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from evaluation.evaluate import write_predictions
from src.common import (load_jsonl, load_split, pick_device, run_dir, save_json, set_seed, technique_pool,
                        technique_texts, tune_thresholds)

BASE = "ehsanaghaei/SecureBERT"
SENT_LEN = 128
TECH_LEN = 256
KS = (3, 5, 10)


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
        width = max(map(len, seqs))
        input_ids = torch.full((len(seqs), width), self.tokenizer.pad_token_id)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return input_ids.to(device), mask.to(device)


@torch.no_grad()
def score_pairs(model, builder, pairs, device, batch_size=128):
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


def load_candidates(run, seed, split):
    """{(doc_id, sent_idx): [(technique, retriever score), ...]} in retriever rank order."""
    path = run_dir(run, "retriever", seed) / f"candidates_{split}.jsonl"
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


def load_model(path, device):
    model = Verifier()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


# ---------------------------------------------------------------- train

def train(args):
    set_seed(args.seed)
    device = pick_device()
    out = run_dir(args.run, "verifier", args.seed)
    pool = technique_pool(args.run, "train")

    text = {(r["doc_id"], r["sent_idx"]): r["text"] for r in load_split("train", args.run)}
    pairs = load_jsonl(out / "pairs_train.jsonl")
    if args.limit:
        pairs = pairs[:args.limit]
    examples = [(text[(p["doc_id"], p["sent_idx"])], p["technique"], float(p["label"])) for p in pairs]
    assert all(t in pool for _, t, _ in examples), "training pair outside the training pool"

    dev = load_split("dev", args.run, args.dev_limit)
    dev_pool = technique_pool(args.run, "dev")
    dev_candidates = load_candidates(args.run, args.seed, "dev")
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
            logits = model(*builder.batch([(s, t) for s, t, _ in batch], device))
            loss = loss_fn(logits, torch.tensor([y for *_, y in batch], device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if args.log_every and (b + 1) % args.log_every == 0:
                print(f"  step {b + 1}/{steps_per_epoch}  loss {np.mean(losses[-args.log_every:]):.4f}"
                      f"  ({time.time() - start:.0f}s)", flush=True)

        dev_best = tune_pipeline(dev, verify(model, builder, dev, dev_candidates, device), dev_pool, args.metric)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **dev_best, "seconds": round(time.time() - start)})
        print(f"epoch {epoch}  loss {np.mean(losses):.4f}  dev {args.metric} {dev_best[args.metric]:.3f}"
              f" (k={dev_best['k']}, theta={dev_best['theta']:.2f}, P {dev_best['micro_precision']:.3f}"
              f" R {dev_best['micro_recall']:.3f})  ({time.time() - start:.0f}s)", flush=True)
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
                                    "retriever": str(run_dir(args.run, "retriever", args.seed)),
                                    "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
                                    "patience": args.patience, "selected_on": f"dev {args.metric}",
                                    "best_dev": best, "limit": args.limit, "pool": pool})
    print(f"best dev {args.metric} {best:.3f} -> {out / 'model.pt'}")


# ---------------------------------------------------------------- predict (the two-stage pipeline)

def predict(args):
    device = pick_device()
    out = run_dir(args.run, "verifier", args.seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = load_model(out / "model.pt", device)
    if args.limit:
        print(f"--limit {args.limit}: predictions cover part of each split, evaluate.py will reject them")

    verified = {}
    for split in ("dev", "test"):
        rows = load_split(split, "full", args.limit)  # every sentence, as evaluate.py expects
        builder = PairBuilder(tokenizer, technique_pool(args.run, split))
        verified[split] = verify(model, builder, rows, load_candidates(args.run, args.seed, split), device)

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
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=2, help="epochs without dev improvement before stopping")
    ap.add_argument("--metric", default="micro_f1", choices=("micro_f1", "macro_f1"),
                    help="dev score for early stopping and for picking k and theta_v")
    ap.add_argument("--log-every", type=int, default=200, help="print the training loss every N steps (0: off)")
    args = ap.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
