"""Checks the scoring math on tiny cases worked out by hand. Run with: make test"""
import pytest

from evaluation.evaluate import evaluate, load_predictions, random_predictions, recall_at_k, restrict, set_scores, write_predictions

LABELS = ["A", "B", "C"]
HIDDEN = {"C"}

# Two reports. s3 has no label. "Z" is out of scope and must be ignored on both sides.
GOLD = {("d1", 1): {"A", "Z"}, ("d1", 2): {"A", "B"}, ("d2", 3): set(), ("d2", 4): {"C"}}
PRED = {("d1", 1): {"A"}, ("d1", 2): {"A"}, ("d2", 3): {"B", "Z"}, ("d2", 4): set()}
SCORES = {
    ("d1", 1): {"A": 0.9, "B": 0.5, "C": 0.1},
    ("d1", 2): {"B": 0.8, "C": 0.6, "A": 0.2},
    ("d2", 3): {"A": 0.3, "B": 0.2, "C": 0.1},
    ("d2", 4): {"A": 0.9, "B": 0.8, "C": 0.7},
}


def preds(predicted, scores=None):
    return {k: {"predicted": v, "scores": scores[k] if scores else None} for k, v in predicted.items()}


def test_micro_and_macro_by_hand():
    s = set_scores(GOLD, PRED, LABELS)
    # Hits: A in s1, A in s2. False alarm: B in s3. Misses: B in s2, C in s4. Z ignored.
    assert s["micro"]["precision"] == pytest.approx(2 / 3)
    assert s["micro"]["recall"] == pytest.approx(2 / 4)
    assert s["micro"]["f1"] == pytest.approx(4 / 7)
    # Per technique F1: A = 1, B = 0, C = 0.
    assert s["macro"]["f1"] == pytest.approx(1 / 3)
    assert s["per_technique"]["B"]["support"] == 1
    assert s["per_technique"]["B"]["predicted"] == 1


def test_full_run_groups():
    report = evaluate(GOLD, preds(PRED), LABELS, HIDDEN)
    assert report["groups"] == {"all": 3, "holdout_trained_on": 1}


def test_withheld_run_groups_and_document_level():
    report = evaluate(GOLD, preds(PRED), LABELS, HIDDEN, withheld_run=True)
    assert report["groups"] == {"all": 3, "seen": 2, "withheld": 1}
    seen = report["sentence_level"]["seen"]["micro"]
    assert seen["precision"] == pytest.approx(2 / 3)
    assert seen["recall"] == pytest.approx(2 / 3)
    # One-technique group: must be 0, not scikit-learn's accuracy-like 0.75.
    assert report["sentence_level"]["withheld"]["micro"]["recall"] == 0.0
    # Reports: d1 gold {A,B} pred {A}; d2 gold {C} pred {B}. 1 hit, 1 false alarm, 2 misses.
    doc = report["document_level"]["all"]["micro"]
    assert doc["precision"] == pytest.approx(1 / 2)
    assert doc["recall"] == pytest.approx(1 / 3)
    assert report["ranking"] == {}  # no scores given


def test_perfect_predictions_score_one():
    perfect = {k: v - {"Z"} for k, v in GOLD.items()}
    report = evaluate(GOLD, preds(perfect), LABELS, HIDDEN, withheld_run=True)
    for group in ("all", "seen", "withheld"):
        assert report["sentence_level"][group]["micro"]["f1"] == pytest.approx(1.0)
        assert report["sentence_level"][group]["macro"]["f1"] == pytest.approx(1.0)


def test_recall_at_k_by_hand():
    # Gold pairs and their rank: (s1,A) 1st, (s2,A) 3rd, (s2,B) 1st, (s4,C) 3rd. Z ignored.
    recalls, pairs = recall_at_k(GOLD, SCORES, LABELS, ks=(1, 3))
    assert pairs == 4
    assert recalls == {"recall@1": pytest.approx(0.5), "recall@3": pytest.approx(1.0)}
    recalls, pairs = recall_at_k(restrict(GOLD, HIDDEN), SCORES, LABELS, ks=(1, 3))
    assert pairs == 1
    assert recalls == {"recall@1": 0.0, "recall@3": 1.0}


def test_missing_sentence_fails_loudly():
    incomplete = {k: v for k, v in PRED.items() if k != ("d2", 3)}
    with pytest.raises(ValueError, match="1 sentences missing"):
        evaluate(GOLD, preds(incomplete), LABELS, HIDDEN)


def test_random_baseline_matches_chance():
    labels = [f"T{i}" for i in range(40)]
    gold = {("d", i): {labels[i % 40]} for i in range(4000)}
    report = evaluate(gold, random_predictions(gold, labels, k=5), labels, set())
    assert report["ranking"]["all"]["recall@5"] == pytest.approx(5 / 40, abs=0.02)


def test_write_and_load_round_trip(tmp_path):
    path = tmp_path / "pred.jsonl"
    write_predictions(path, [{"doc_id": k[0], "sent_idx": k[1], "predicted": PRED[k], "scores": SCORES[k]} for k in PRED])
    loaded = load_predictions(path)
    assert loaded[("d2", 3)]["predicted"] == {"B", "Z"}
    assert loaded[("d1", 1)]["scores"]["A"] == pytest.approx(0.9)
