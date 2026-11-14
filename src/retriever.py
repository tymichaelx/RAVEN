"""Stage 1: the retriever (a bi-encoder).

Sentence and technique text are encoded separately by the same MiniLM model, mean
pooled, passed through a linear projection and normalized. The score of a
(sentence, technique) pair is the cosine similarity of the two vectors.

Training uses a multi-positive contrastive loss: each true technique of a sentence
competes only against the techniques the sentence is not labeled with, never against
its other true labels. Unlabeled sentences don't enter the loss.

On its own, the retriever is the single-stage system: predict the top-k techniques
whose score is at least theta. k and theta are tuned on dev.

Usage:
  python -m src.retriever train   --run full --seed 0
  python -m src.retriever predict --run full --seed 0
  (add --smoke for a quick check that the code runs; writes to runs/smoke/retriever/)

Outputs in runs/<run>/retriever/seed<N>/:
  model.pt, config.json, history.json      training
  thresholds.json                          k and theta picked on dev
  dev.jsonl, test.jsonl                    predictions for evaluation/evaluate.py
  candidates_{train,dev,test}.jsonl        top 10 techniques per sentence, the verifier's input
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from evaluation.evaluate import recall_at_k, write_predictions
from src.common import (SPLITS, add_smoke_option, apply_thresholds, load_split, matrix, model_dir, pick_device, save_json,
                        set_seed, technique_pool, technique_texts, tune_thresholds, write_jsonl)

BASE = "sentence-transformers/all-MiniLM-L6-v2"
SENT_LEN = 128
TECH_LEN = 256
K_MAX = 10           # candidates handed to the verifier
KS = (3, 5, 10)      # k values tried for the single-stage system


class Retriever(nn.Module):
    def __init__(self, base=BASE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        d = self.encoder.config.hidden_size
        self.proj = nn.Linear(d, d)
        # Start as the identity, so training begins from the pretrained sentence space.
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids, attention_mask, **_):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(self.proj(pooled), dim=-1)


def multi_positive_loss(sims, positives, tau):
    """Contrastive loss where each true technique competes only with the sentence's negatives.

    sims: [B, T] cosine similarities. positives: [B, T] bool, True where t is a gold label.
    For each gold pair (i, p): -log( e^{s_ip/tau} / (e^{s_ip/tau} + sum over negatives n of e^{s_in/tau}) ).
    The sentence's other gold techniques are left out of the denominator.
    """
    logits = sims / tau
    negatives_lse = torch.logsumexp(logits.masked_fill(positives, float("-inf")), dim=1, keepdim=True)
    per_pair = torch.logaddexp(logits, negatives_lse) - logits
    return per_pair[positives].mean()


def tokenize(tokenizer, texts, max_len, device):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)


@torch.no_grad()
def encode(model, tokenizer, texts, max_len, device, batch_size=128):
    model.eval()
    out = [model(**tokenize(tokenizer, texts[i:i + batch_size], max_len, device)) for i in range(0, len(texts), batch_size)]
    return torch.cat(out)


@torch.no_grad()
def rank(model, tokenizer, rows, pool, device):
    """{(doc_id, sent_idx): {technique: cosine}} for every sentence and every pool technique."""
    texts = technique_texts(pool)
    tech = encode(model, tokenizer, [texts[t] for t in pool], TECH_LEN, device)
    sent = encode(model, tokenizer, [r["text"] for r in rows], SENT_LEN, device)
    sims = (sent @ tech.T).float().cpu().numpy()
    return {(r["doc_id"], r["sent_idx"]): dict(zip(pool, map(float, sims[i]))) for i, r in enumerate(rows)}


def dev_recall(model, tokenizer, dev, pool, device):
    gold = {(r["doc_id"], r["sent_idx"]): set(r["labels"]) for r in dev if r["labels"]}
    labeled = [r for r in dev if r["labels"]]
    recalls, _ = recall_at_k(gold, rank(model, tokenizer, labeled, pool, device), pool, ks=(1, 3, 5, 10))
    return recalls


def load_model(path, device):
    model = Retriever()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


# ---------------------------------------------------------------- train

def train(args):
    set_seed(args.seed)
    device = pick_device()
    out = model_dir(args, "retriever")
    pool = technique_pool(args.run, "train")
    texts = technique_texts(pool)
    col = {t: j for j, t in enumerate(pool)}

    labeled = [r for r in load_split("train", args.run, args.limit) if r["labels"]]
    dev = load_split("dev", args.run, args.limit)
    print(f"{args.run} run | {len(labeled)} labeled training sentences | {len(pool)} techniques | {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = Retriever().to(device)
    tech_batch = tokenize(tokenizer, [texts[t] for t in pool], TECH_LEN, device)

    steps_per_epoch = math.ceil(len(labeled) / args.batch_size)
    total = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    rng = np.random.default_rng(args.seed)

    select = f"recall@{args.select_k}"
    best, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, losses = time.time(), []
        order = rng.permutation(len(labeled))
        for b in range(steps_per_epoch):
            batch = [labeled[i] for i in order[b * args.batch_size:(b + 1) * args.batch_size]]
            positives = torch.zeros(len(batch), len(pool), dtype=torch.bool)
            for i, r in enumerate(batch):
                positives[i, [col[t] for t in r["labels"]]] = True
            sent = model(**tokenize(tokenizer, [r["text"] for r in batch], SENT_LEN, device))
            tech = model(**tech_batch)
            loss = multi_positive_loss(sent @ tech.T, positives.to(device), args.tau)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        recalls = dev_recall(model, tokenizer, dev, pool, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **recalls, "seconds": round(time.time() - start)})
        print(f"epoch {epoch:2}  loss {np.mean(losses):.4f}  dev " +
              "  ".join(f"{k} {v:.3f}" for k, v in recalls.items()) + f"  ({time.time() - start:.0f}s)")
        if recalls[select] > best:
            best, bad_epochs = recalls[select], 0
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"no dev improvement for {args.patience} epochs, stopping")
                break

    save_json(out / "history.json", history)
    save_json(out / "config.json", {"model": "retriever", "base": BASE, "run": args.run, "seed": args.seed,
                                    "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
                                    "tau": args.tau, "patience": args.patience, "selected_on": select,
                                    "best_dev": best, "limit": args.limit, "pool": pool})
    print(f"best dev {select} {best:.3f} -> {out / 'model.pt'}")


# ---------------------------------------------------------------- predict

def prediction_rows(split, run):
    """Sentences to score. Train follows the run's filtering (it feeds the verifier's training);
    dev and test keep every sentence, because evaluate.py checks that none is missing."""
    return load_split(split, run if split == "train" else "full")


def predict(args):
    device = pick_device()
    out = model_dir(args, "retriever")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = load_model(out / "model.pt", device)

    scores = {}
    for split in SPLITS:
        pool = technique_pool(args.run, split)
        rows = prediction_rows(split, args.run)
        scores[split] = rank(model, tokenizer, rows, pool, device)
        write_jsonl(out / f"candidates_{split}.jsonl", [
            {"doc_id": r["doc_id"], "sent_idx": r["sent_idx"],
             "candidates": [[t, s] for t, s in sorted(scores[split][(r["doc_id"], r["sent_idx"])].items(),
                                                      key=lambda x: (-x[1], x[0]))[:K_MAX]]}
            for r in rows])

    # Tune k and theta on dev, with the dev sentences and labels this run trains with.
    dev_pool = technique_pool(args.run, "dev")
    S, G = matrix(load_split("dev", args.run), scores["dev"], dev_pool)
    best = tune_thresholds(S, G, ks=KS, metric=args.metric)
    save_json(out / "thresholds.json", {"tuned_on": "dev", "metric": args.metric, **best})
    print(f"dev: k={best['k']} theta={best['theta']:.3f}  micro F1 {best['micro_f1']:.3f}  macro F1 {best['macro_f1']:.3f}"
          f"  (P {best['micro_precision']:.3f} R {best['micro_recall']:.3f})")

    for split in ("dev", "test"):
        pool = technique_pool(args.run, split)
        rows = prediction_rows(split, args.run)
        S, _ = matrix(rows, scores[split], pool)
        keep = apply_thresholds(S, best["k"], best["theta"])
        write_predictions(out / f"{split}.jsonl", [
            {"doc_id": r["doc_id"], "sent_idx": r["sent_idx"],
             "predicted": [pool[j] for j in np.flatnonzero(keep[i])],
             "scores": scores[split][(r["doc_id"], r["sent_idx"])]}
            for i, r in enumerate(rows)])
    print(f"predictions -> {out}/dev.jsonl, {out}/test.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("train", "predict"))
    ap.add_argument("--run", default="full", choices=("full", "withheld"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only the first N train and dev sentences (smoke runs)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tau", type=float, default=0.05, help="temperature of the contrastive loss")
    ap.add_argument("--patience", type=int, default=3, help="epochs without dev improvement before stopping")
    ap.add_argument("--select-k", type=int, default=5, help="early stopping watches dev recall@k")
    ap.add_argument("--metric", default="micro_f1", choices=("micro_f1", "macro_f1"),
                    help="dev score that picks k and theta")
    add_smoke_option(ap)
    args = ap.parse_args()
    if args.smoke:
        args.run, args.limit, args.epochs = "full", args.limit or 400, 1
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
