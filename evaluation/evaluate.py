"""Score a model's predictions. Every model in the project is graded by this file.

A model writes one JSON line per sentence of the split (use `write_predictions`):

  {"doc_id": "...", "sent_idx": 3, "predicted": ["T1059"], "scores": {"T1059": 0.91, ...}}

`predicted` is the final answer, possibly empty. `scores` is optional: a score per
technique, higher meaning more likely; it enables recall@k and average precision.

Only the 40 in-scope techniques count; other labels are ignored on both sides.

Groups reported, depending on how the model was trained:
  full run (default), trained on all 40:      all, holdout_trained_on (the 10, trained on)
  withheld run (--withheld-run), without 10:  all, seen (30), withheld (10)
Hypothesis 3 compares "withheld" with "holdout_trained_on": the same 10 techniques.

Usage:
  python -m evaluation.evaluate --pred runs/retriever/test.jsonl [--split dev] [--withheld-run]
  python -m evaluation.evaluate --random-baseline [--k 5]
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.holdout import load_hidden

DATA = Path("data/extracted")
KS = (1, 3, 5, 10)


# ---------------------------------------------------------------- files

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def in_scope_labels():
    label_map = json.load(open(DATA / "label_map.json"))
    return sorted(tid for tid, info in label_map.items() if info["in_scope"])


def load_gold(split):
    """{(doc_id, sent_idx): set of technique ids}."""
    return {(r["doc_id"], r["sent_idx"]): set(r["labels"]) for r in load_jsonl(DATA / f"{split}.jsonl")}


def load_predictions(path):
    """{(doc_id, sent_idx): {"predicted": set, "scores": dict or None}}."""
    preds = {}
    for r in load_jsonl(path):
        key = (r["doc_id"], r["sent_idx"])
        if key in preds:
            raise ValueError(f"Sentence {key} appears twice in {path}")
        preds[key] = {"predicted": set(r["predicted"]), "scores": r.get("scores")}
    return preds


def write_predictions(path, rows):
    """rows: dicts with doc_id, sent_idx, predicted and optionally scores."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            out = {"doc_id": r["doc_id"], "sent_idx": r["sent_idx"], "predicted": sorted(r["predicted"])}
            if r.get("scores") is not None:
                out["scores"] = {t: float(s) for t, s in r["scores"].items()}
            f.write(json.dumps(out) + "\n")


# ---------------------------------------------------------------- metrics

def to_matrix(keys, label_sets, labels):
    """Binary matrix: one row per sentence, one column per technique in `labels`."""
    col = {t: j for j, t in enumerate(labels)}
    m = np.zeros((len(keys), len(labels)), dtype=int)
    for i, key in enumerate(keys):
        for t in label_sets[key]:
            if t in col:
                m[i, col[t]] = 1
    return m


def prf(tp, fp, fn):
    """Precision, recall and F1 from counts. An empty denominator gives 0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def set_scores(gold, predicted, labels):
    """Micro, macro and per-technique precision/recall/F1 over `labels`.

    Computed from counts on purpose: for a one-technique group scikit-learn's micro
    average also counts correct "no" answers, which turns it into accuracy.
    """
    keys = sorted(gold)
    y_true = to_matrix(keys, gold, labels)
    y_pred = to_matrix(keys, predicted, labels)
    tp = (y_true & y_pred).sum(axis=0)
    fp = ((1 - y_true) & y_pred).sum(axis=0)
    fn = (y_true & (1 - y_pred)).sum(axis=0)
    per_technique = {
        t: {**prf(tp[j], fp[j], fn[j]), "support": int(tp[j] + fn[j]), "predicted": int(tp[j] + fp[j])}
        for j, t in enumerate(labels)
    }
    return {
        "micro": prf(tp.sum(), fp.sum(), fn.sum()),
        "macro": {m: float(np.mean([s[m] for s in per_technique.values()])) for m in ("precision", "recall", "f1")},
        "per_technique": per_technique,
    }


def by_document(label_sets):
    """A report's labels are the union of its sentences' labels."""
    docs = defaultdict(set)
    for (doc_id, _), s in label_sets.items():
        docs[doc_id] |= s
    return dict(docs)


def recall_at_k(gold, scores, labels, ks=KS):
    """Share of gold (sentence, technique) pairs whose technique is in the top k.

    Ranks over all of `labels`; only gold labels in `labels` are counted. To score one
    group, pass gold filtered to that group with `restrict`.
    """
    hits = dict.fromkeys(ks, 0)
    total = 0
    for key, gold_set in gold.items():
        gold_set = gold_set & set(labels)
        if not gold_set:
            continue
        order = sorted((t for t in scores[key] if t in labels), key=lambda t: (-scores[key][t], t))
        position = {t: i for i, t in enumerate(order)}
        for t in gold_set:
            total += 1
            for k in ks:
                hits[k] += position.get(t, len(labels)) < k
    return {f"recall@{k}": hits[k] / total if total else 0.0 for k in ks}, total


def average_precision(gold, scores, labels):
    """Area under the precision-recall curve over every (sentence, technique) decision."""
    keys = sorted(gold)
    y_true = to_matrix(keys, gold, labels).ravel()
    if y_true.sum() == 0:
        return 0.0
    lowest = min(min(s.values(), default=0.0) for s in scores.values()) - 1  # for unscored techniques
    y_score = np.array([[scores[key].get(t, lowest) for t in labels] for key in keys]).ravel()
    return float(average_precision_score(y_true, y_score))


def restrict(label_sets, group):
    return {key: s & set(group) for key, s in label_sets.items()}


# ---------------------------------------------------------------- report

def technique_groups(labels, hidden, withheld_run):
    holdout = sorted(set(labels) & hidden)
    if withheld_run:
        groups = {"all": sorted(labels), "seen": sorted(set(labels) - hidden), "withheld": holdout}
    else:
        groups = {"all": sorted(labels), "holdout_trained_on": holdout}
    return {name: g for name, g in groups.items() if g}


def evaluate(gold, preds, labels, hidden, withheld_run=False):
    """Every metric for one predictions file, as a JSON-ready dict."""
    missing, extra = gold.keys() - preds.keys(), preds.keys() - gold.keys()
    if missing or extra:
        raise ValueError(f"Predictions don't match the split: {len(missing)} sentences missing, "
                         f"{len(extra)} not in the split")
    predicted = {key: p["predicted"] for key, p in preds.items()}
    scores = {key: p["scores"] for key, p in preds.items()}
    has_scores = all(s is not None for s in scores.values())

    groups = technique_groups(labels, hidden, withheld_run)
    report = {"run": "withheld" if withheld_run else "full", "sentences": len(gold),
              "groups": {name: len(g) for name, g in groups.items()},
              "sentence_level": {}, "document_level": {}, "ranking": {}}
    gold_docs, pred_docs = by_document(gold), by_document(predicted)
    for name, group in groups.items():
        report["sentence_level"][name] = set_scores(gold, predicted, group)
        document = set_scores(gold_docs, pred_docs, group)
        report["document_level"][name] = {"micro": document["micro"], "macro": document["macro"]}
        if has_scores:
            recalls, pairs = recall_at_k(restrict(gold, group), scores, labels)
            report["ranking"][name] = {**recalls, "average_precision": average_precision(gold, scores, group),
                                       "gold_pairs": pairs}
    return report


def random_predictions(gold, labels, k, seed=0):
    """Rank techniques randomly for each sentence and predict the top k: the chance floor."""
    rng = random.Random(seed)
    preds = {}
    for key in sorted(gold):
        order = rng.sample(labels, len(labels))
        preds[key] = {"predicted": set(order[:k]), "scores": {t: -i for i, t in enumerate(order)}}
    return preds


def print_report(report):
    w = 2 + max(len(n) for n in report["groups"])
    print(f"{report['run']} run | {report['sentences']} sentences | groups: "
          + ", ".join(f"{n} ({c})" for n, c in report["groups"].items()))
    for level, title in (("sentence_level", "Sentence level"), ("document_level", "Document level")):
        print(f"\n{title:<{w}}  micro P  micro R  micro F1 | macro P  macro R  macro F1")
        for name, s in report[level].items():
            values = [s["micro"]["precision"], s["micro"]["recall"], s["micro"]["f1"],
                      s["macro"]["precision"], s["macro"]["recall"], s["macro"]["f1"]]
            print(f"  {name:<{w - 2}}  " + "  ".join(f"{v:7.3f}" for v in values[:3])
                  + " | " + "  ".join(f"{v:7.3f}" for v in values[3:]))
    if report["ranking"]:
        print(f"\n{'Ranking':<{w}}  " + "  ".join(f"recall@{k:<2}" for k in KS) + "  avg prec  gold pairs")
        for name, r in report["ranking"].items():
            print(f"  {name:<{w - 2}}  " + "  ".join(f"{r[f'recall@{k}']:9.3f}" for k in KS)
                  + f"  {r['average_precision']:8.3f}  {r['gold_pairs']:10}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--pred", type=Path, help="predictions JSONL to score")
    what.add_argument("--random-baseline", action="store_true", help="score random rankings (withheld groups)")
    ap.add_argument("--withheld-run", action="store_true", help="model was trained without the 10 holdout techniques")
    ap.add_argument("--split", default="test", choices=("train", "dev", "test"))
    ap.add_argument("--k", type=int, default=5, help="random baseline: techniques predicted per sentence")
    ap.add_argument("--out", type=Path, help="also save the full report as JSON")
    args = ap.parse_args()

    labels, hidden, gold = in_scope_labels(), load_hidden(), load_gold(args.split)
    if args.pred:
        report = evaluate(gold, load_predictions(args.pred), labels, hidden, args.withheld_run)
    else:
        report = evaluate(gold, random_predictions(gold, labels, args.k), labels, hidden, withheld_run=True)
    print_report(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
