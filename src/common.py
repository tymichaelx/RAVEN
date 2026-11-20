"""Shared pieces for the retriever and the verifier.

Both models go through these functions, so they see the same sentences, the same
technique texts and the same candidate pool. That keeps the two stages consistent.

Run types:
  full      train and test on all 40 in-scope techniques (hypotheses 1 and 2)
  withheld  train and tune without the 10 holdout techniques; add them back for test (hypothesis 3)
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.holdout import apply_holdout, candidate_techniques, load_hidden

DATA = Path("data/extracted")
TECHNIQUES = Path("data/techniques.json")
RUNS = Path("runs")
RUN_TYPES = ("full", "withheld")
SPLITS = ("train", "dev", "test")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def in_scope_ids():
    """The 40 techniques that are scored (at least 20 training sentences)."""
    label_map = json.load(open(DATA / "label_map.json"))
    return sorted(tid for tid, info in label_map.items() if info["in_scope"])


def technique_pool(run, split):
    """Technique ids the model may rank for this run and split, sorted.

    Full run: the 40 in-scope techniques everywhere.
    Withheld run: the 30 kept techniques for train and dev; all 40 for test.
    """
    if run not in RUN_TYPES:
        raise ValueError(f"run must be one of {RUN_TYPES}, got {run!r}")
    ids = {tid: None for tid in in_scope_ids()}
    if run == "withheld" and split != "test":
        ids = candidate_techniques(ids, load_hidden())
    return sorted(ids)


def technique_texts(ids):
    """{id: "<name>: <first paragraph>"}: the one text both models read for a technique."""
    techniques = json.load(open(TECHNIQUES))["techniques"]
    return {tid: techniques[tid]["text"] for tid in ids}


def load_split(split, run, limit=None):
    """Sentences of a split as the model should see them.

    Withheld run, train and dev: apply_holdout removes the 10 withheld techniques.
    Test is never filtered. `labels` is then cut down to the technique pool, so a
    sentence whose only labels are rare out-of-scope techniques counts as unlabeled.
    `gold` keeps the full label list. `limit` keeps the first N sentences (smoke runs).
    """
    rows = load_jsonl(DATA / f"{split}.jsonl")
    if run == "withheld" and split != "test":
        rows = apply_holdout(rows, load_hidden())
    if limit:
        rows = rows[:limit]
    pool = set(technique_pool(run, split))
    return [{**r, "gold": r["labels"], "labels": [t for t in r["labels"] if t in pool]} for r in rows]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_dir(run, model, seed):
    """runs/<run>/<model>/seed<N>/ (gitignored)."""
    return RUNS / run / model / f"seed{seed}"


def add_smoke_option(parser):
    parser.add_argument("--smoke", action="store_true",
                        help="quick check that the code runs: a few hundred sentences, 1 epoch, "
                             "written to runs/smoke/<model>/. The scores mean nothing.")


def model_dir(args, model):
    """Folder for a model's files: runs/<run>/<model>/seed<N>/, or runs/smoke/<model>/ with --smoke."""
    if args.smoke:
        return RUNS / "smoke" / model
    return run_dir(args.run, model, args.seed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


# ---------------------------------------------------------------- thresholds

def matrix(rows, scores, pool):
    """Score matrix S [sentences, techniques] and gold 0/1 matrix G, columns in pool order.

    rows: sentences with doc_id, sent_idx, labels. scores: {(doc_id, sent_idx): {tid: score}}.
    A technique missing from a sentence's scores gets -inf (never predicted).
    """
    S = np.full((len(rows), len(pool)), -np.inf)
    G = np.zeros((len(rows), len(pool)), dtype=bool)
    col = {t: j for j, t in enumerate(pool)}
    for i, r in enumerate(rows):
        for t, s in scores[(r["doc_id"], r["sent_idx"])].items():
            if t in col:
                S[i, col[t]] = s
        for t in r["labels"]:
            if t in col:
                G[i, col[t]] = True
    return S, G


def f1_counts(P, G):
    """Micro and macro precision/recall/F1 of a 0/1 prediction matrix (same math as evaluate.py)."""
    tp, fp, fn = (P & G).sum(0), (P & ~G).sum(0), (~P & G).sum(0)

    def prf(tp, fp, fn):
        p = np.divide(tp, tp + fp, out=np.zeros(np.shape(tp)), where=(tp + fp) > 0)
        r = np.divide(tp, tp + fn, out=np.zeros(np.shape(tp)), where=(tp + fn) > 0)
        f = np.divide(2 * p * r, p + r, out=np.zeros(np.shape(tp)), where=(p + r) > 0)
        return p, r, f

    mp, mr, mf = prf(np.array(tp.sum(), float), np.array(fp.sum(), float), np.array(fn.sum(), float))
    cp, cr, cf = prf(tp.astype(float), fp.astype(float), fn.astype(float))
    return {"micro_precision": float(mp), "micro_recall": float(mr), "micro_f1": float(mf),
            "macro_precision": float(cp.mean()), "macro_recall": float(cr.mean()), "macro_f1": float(cf.mean())}


def top_k_mask(S, k):
    """True where a technique is among a sentence's k highest scores (ties: earlier column first)."""
    if k is None:
        return np.isfinite(S)
    order = np.argsort(-S, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(S.shape[0])[:, None], order] = np.arange(S.shape[1])
    return (rank < k) & np.isfinite(S)


def tune_thresholds(S, G, ks=(None,), metric="micro_f1", steps=200):
    """Pick k and a score threshold that maximise `metric` on dev. Use dev only.

    Predicts technique t for a sentence when t is in its top k and its score >= threshold.
    k=None means no top-k cut. Returns the best setting with its dev scores.
    """
    finite = S[np.isfinite(S)]
    thetas = np.unique(np.concatenate([np.linspace(finite.min(), finite.max(), steps + 1), [-np.inf]]))
    best = None
    for k in ks:
        top = top_k_mask(S, k)
        for theta in thetas:
            result = {"k": k, "theta": float(theta), **f1_counts(top & (S >= theta), G)}
            if best is None or result[metric] > best[metric]:
                best = result
    return best


def apply_thresholds(S, k, theta):
    return top_k_mask(S, k) & (S >= theta)
