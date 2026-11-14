"""Build the verifier's training pairs from the trained retriever's mistakes on train.

Pairs (sentence, technique, label):
  gold           every gold technique of a sentence, label 1 (whether or not it was retrieved)
  labeled_neg    a labeled sentence's top-ranked techniques that are not gold, label 0
  unlabeled_neg  an unlabeled sentence's top-ranked techniques, label 0, so the verifier
                 learns to reject everything for sentences that describe no technique

Guard: TRAM misses some true labels. A non-gold candidate that the retriever scores at
least as high as a typical true pair (default: the 75th percentile of gold-pair scores)
may be a missed label, so it is left out of training and written to spot_check.csv for
a look by hand.

Reads  runs/<run>/retriever/seed<N>/candidates_train.jsonl  (from `python -m src.retriever predict`)
Writes runs/<run>/verifier/seed<N>/pairs_train.jsonl, spot_check.csv, mining.json

Usage:
  python -m src.mine_negatives --run full --seed 0
"""
import argparse
import csv
import json

import numpy as np

from src.common import add_smoke_option, load_jsonl, load_split, model_dir, save_json, technique_pool, technique_texts, write_jsonl


def mine(rows, candidates, neg_k, unlabeled_k, unlabeled_frac, guard_quantile, seed):
    """Returns (pairs, skipped, summary). rows and candidates are keyed by (doc_id, sent_idx)."""
    gold_scores = [s for key, r in rows.items() for t, s in candidates[key] if t in r["labels"]]
    guard = float(np.quantile(gold_scores, guard_quantile))
    rng = np.random.default_rng(seed)

    pairs, skipped = [], []
    for key, r in rows.items():
        cands = candidates[key]
        score = dict(cands)
        if r["labels"]:
            for t in r["labels"]:
                pairs.append({"key": key, "technique": t, "label": 1, "source": "gold", "retriever_score": score.get(t)})
            negatives = [(t, s) for t, s in cands if t not in r["labels"]][:neg_k]
            source = "labeled_neg"
        else:
            if rng.random() >= unlabeled_frac:
                continue
            negatives = cands[:unlabeled_k]
            source = "unlabeled_neg"
        for t, s in negatives:
            pair = {"key": key, "technique": t, "label": 0, "source": source, "retriever_score": s}
            (skipped if s >= guard else pairs).append(pair)

    counts = {src: sum(p["source"] == src for p in pairs) for src in ("gold", "labeled_neg", "unlabeled_neg")}
    summary = {"guard": guard, "guard_quantile": guard_quantile, "pairs": len(pairs), **counts,
               "skipped_by_guard": len(skipped), "neg_k": neg_k, "unlabeled_k": unlabeled_k,
               "unlabeled_frac": unlabeled_frac, "seed": seed}
    return pairs, skipped, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="full", choices=("full", "withheld"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neg-k", type=int, default=5,
                    help="negatives per labeled sentence, from its top candidates (ranks below 5 are easy on train)")
    ap.add_argument("--unlabeled-k", type=int, default=3, help="negatives per sampled unlabeled sentence")
    ap.add_argument("--unlabeled-frac", type=float, default=0.4, help="share of unlabeled sentences used")
    ap.add_argument("--guard-quantile", type=float, default=0.75,
                    help="skip negatives scoring above this quantile of gold-pair scores")
    ap.add_argument("--limit", type=int, help="first N train sentences only (smoke runs)")
    add_smoke_option(ap)
    args = ap.parse_args()
    if args.smoke:
        args.run, args.limit = "full", args.limit or 400

    source = model_dir(args, "retriever")
    out = model_dir(args, "verifier")
    rows = {(r["doc_id"], r["sent_idx"]): r for r in load_split("train", args.run, args.limit)}
    candidates = {(c["doc_id"], c["sent_idx"]): [tuple(x) for x in c["candidates"]]
                  for c in load_jsonl(source / "candidates_train.jsonl")}
    candidates = {k: candidates[k] for k in rows}

    pairs, skipped, summary = mine(rows, candidates, args.neg_k, args.unlabeled_k, args.unlabeled_frac,
                                   args.guard_quantile, args.seed)
    pool = set(technique_pool(args.run, "train"))
    assert all(p["technique"] in pool for p in pairs), "a pair uses a technique outside the training pool"

    write_jsonl(out / "pairs_train.jsonl", [
        {"doc_id": p["key"][0], "sent_idx": p["key"][1], "technique": p["technique"], "label": p["label"],
         "source": p["source"], "retriever_score": p["retriever_score"]} for p in pairs])
    names = technique_texts(sorted(pool))
    with open(out / "spot_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["retriever_score", "technique", "technique_name", "sentence_labels", "sentence", "really_a_match?"])
        for p in sorted(skipped, key=lambda p: -p["retriever_score"]):
            r = rows[p["key"]]
            w.writerow([f"{p['retriever_score']:.3f}", p["technique"], names[p["technique"]].split(":")[0],
                        " ".join(r["labels"]), r["text"], ""])
    save_json(out / "mining.json", {"retriever": str(source), "run": args.run, **summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
