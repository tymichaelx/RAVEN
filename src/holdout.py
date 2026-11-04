"""Pick 10 techniques to hide while the model is training.

Writes data/holdout.json. That file is the list training and testing should use.
Run it once and keep the file. Do not draw a new list later.

A technique is eligible when label_map says it is in_scope (at least 20 training
sentences) and it has at least 5 test sentences. The rare techniques marked
in_scope false stay out of this list.

Rule, fixed before training:
  1. Sort the eligible techniques by how many training sentences they have.
     The bottom third is low, the middle third is mid, the top third is high.
     Extra techniques, when the count is not divisible by 3, go to the lower thirds.
  2. Walk tactics A to Z. From each tactic take one technique that has not been
     taken yet, rotating low, then mid, then high. If that tactic has nothing
     left in the wanted third, take the next third.
  3. Stop at 10. When several techniques in a tactic fit the third, take the one
     closest to the middle of that third. A tie goes to the smaller technique id.

Train and dev: if every label on a sentence is one of these 10, drop the sentence.
If the sentence also has a label we are keeping, drop only the hidden label.
Test sentences stay as they are, so the hidden techniques can be scored.
Their descriptions stay out of the candidate pool until test time.

Usage:
  python src/holdout.py          # pick the list and write data/holdout.json

In training code:
  from src.holdout import load_hidden, apply_holdout, candidate_techniques
  hidden = load_hidden()
  train = apply_holdout(train, hidden)               # same for dev, never for test
  pool = candidate_techniques(techniques, hidden)    # descriptions the model may see
"""
import argparse
import json
from pathlib import Path
from statistics import median

BANDS = ("low", "mid", "high")
N_HOLDOUT = 10
MIN_TEST = 5


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def band_sizes(n):
    """Split n items into 3 groups, as even as possible. Extras go to the lower groups."""
    base, extra = divmod(n, 3)
    return [base + (1 if i < extra else 0) for i in range(3)]


def eligible_techniques(label_map, techniques):
    candidates = []
    for tid, info in label_map.items():
        if not info["in_scope"] or info["test"] < MIN_TEST:
            continue
        if tid not in techniques:
            raise SystemExit(f"{tid} is eligible but has no entry in techniques.json")
        candidates.append({
            "id": tid,
            "name": info["name"],
            "tactics": list(techniques[tid]["tactics"]),
            "train": info["train"],
            "dev": info["dev"],
            "test": info["test"],
        })
    candidates.sort(key=lambda c: (c["train"], c["id"]))
    sizes = band_sizes(len(candidates))
    start = 0
    for band, size in zip(BANDS, sizes):
        for candidate in candidates[start:start + size]:
            candidate["band"] = band
        start += size
    return candidates, dict(zip(BANDS, sizes))


def choose(candidates):
    """One technique per tactic, rotating frequency thirds, until N_HOLDOUT."""
    band_median = {
        band: median(c["train"] for c in candidates if c["band"] == band)
        for band in BANDS
    }
    tactics = sorted({t for c in candidates for t in c["tactics"]})
    chosen = []
    taken = set()
    # `step` counts picks so far and decides which third we want next
    # (low, mid, high, low, ...). `shift` is the fallback: if the tactic has
    # nothing left in the wanted third, try the next third, then the one after.
    step = 0
    while len(chosen) < N_HOLDOUT:
        progress = False
        for tactic in tactics:
            if len(chosen) >= N_HOLDOUT:
                break
            pick = None
            for shift in range(3):
                band = BANDS[(step + shift) % 3]
                fits = [
                    c for c in candidates
                    if c["band"] == band and tactic in c["tactics"] and c["id"] not in taken
                ]
                if fits:
                    fits.sort(key=lambda c: (abs(c["train"] - band_median[band]), c["id"]))
                    pick = fits[0]
                    break
            if pick is None:
                continue
            chosen.append({**pick, "picked_for": tactic})
            taken.add(pick["id"])
            step += 1
            progress = True
        if not progress:
            break
    if len(chosen) < N_HOLDOUT:
        raise SystemExit(f"Could only choose {len(chosen)} of {N_HOLDOUT} techniques")
    return chosen


# Labels in data/extracted are already parent techniques (build_dataset.py turns
# T1003.001 into T1003), so they can be compared with the hidden ids directly.

def split_effect(sentences, hidden):
    """How train/dev filtering would treat this split. Does not edit the sentences."""
    dropped = 0
    stripped_labels = 0
    sentences_touched = 0
    for sentence in sentences:
        labels = sentence["labels"]
        kept = [label for label in labels if label not in hidden]
        removed = len(labels) - len(kept)
        if labels and not kept:
            dropped += 1
        elif removed:
            sentences_touched += 1
            stripped_labels += removed
    return {
        "sentences": len(sentences),
        "dropped": dropped,
        "sentences_with_a_hidden_label_removed": sentences_touched,
        "hidden_labels_removed": stripped_labels,
        "kept": len(sentences) - dropped,
    }


def test_effect(sentences, hidden):
    """Test is not filtered. Count what the hidden techniques contribute there."""
    with_hidden = 0
    hidden_labels = 0
    for sentence in sentences:
        n = sum(1 for label in sentence["labels"] if label in hidden)
        if n:
            with_hidden += 1
            hidden_labels += n
    return {
        "sentences": len(sentences),
        "sentences_with_a_hidden_label": with_hidden,
        "hidden_label_count": hidden_labels,
    }


def load_hidden(path=Path("data/holdout.json")):
    """The withheld technique ids, as a set."""
    with open(path) as f:
        return set(json.load(f)["ids"])


def apply_holdout(sentences, hidden):
    """Remove the withheld techniques from train or dev sentences.

    A sentence whose labels are all withheld is dropped. Otherwise only the
    withheld labels are removed. Unlabeled sentences stay. Returns new dicts and
    leaves the input untouched. Do not call this on the test split.
    """
    kept_sentences = []
    for sentence in sentences:
        labels = sentence["labels"]
        kept = [label for label in labels if label not in hidden]
        if labels and not kept:
            continue
        kept_sentences.append({
            **sentence,
            "labels": kept,
            # raw_labels keep sub-technique ids (T1003.001), so compare the parent part.
            "raw_labels": [r for r in sentence["raw_labels"] if r.split(".")[0] not in hidden],
        })
    return kept_sentences


def candidate_techniques(techniques, hidden):
    """The technique descriptions allowed during training: everything except the withheld ones."""
    return {tid: info for tid, info in techniques.items() if tid not in hidden}


def build(label_map, techniques, splits):
    candidates, sizes = eligible_techniques(label_map, techniques)
    chosen = choose(candidates)
    hidden = {t["id"] for t in chosen}
    covered = sorted({t for tech in chosen for t in tech["tactics"]})
    return {
        "purpose": (
            "Ten techniques withheld from training so the model can be tested "
            "on methods it never saw. Drawn from techniques with enough examples, "
            "not from the rare techniques marked in_scope false."
        ),
        "ids": [t["id"] for t in chosen],
        "pool": {
            "eligible": len(candidates),
            "min_test_sentences": MIN_TEST,
            "band_sizes": sizes,
        },
        "techniques": chosen,
        "tactics_covered": covered,
        "train_and_dev": (
            "Drop a sentence when every label on it is withheld. "
            "If it also has a label that is kept, remove only the withheld label. "
            "Leave test sentences unchanged. "
            "Keep the withheld descriptions out of the candidate pool until test time."
        ),
        "effects": {
            "train": split_effect(splits["train"], hidden),
            "dev": split_effect(splits["dev"], hidden),
            "test": test_effect(splits["test"], hidden),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("data/extracted/label_map.json"))
    ap.add_argument("--techniques", type=Path, default=Path("data/techniques.json"))
    ap.add_argument("--splits", type=Path, default=Path("data/extracted"))
    ap.add_argument("--out", type=Path, default=Path("data/holdout.json"))
    args = ap.parse_args()

    label_map = json.load(open(args.labels))
    techniques = json.load(open(args.techniques))["techniques"]
    splits = {name: load_jsonl(args.splits / f"{name}.jsonl") for name in ("train", "dev", "test")}

    report = build(label_map, techniques, splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"{len(report['ids'])} withheld techniques -> {args.out}")
    for tech in report["techniques"]:
        print(f"  {tech['id']}  {tech['band']:4}  {tech['name']}")
    train = report["effects"]["train"]
    print(f"train sentences dropped: {train['dropped']} of {train['sentences']}")


if __name__ == "__main__":
    main()
