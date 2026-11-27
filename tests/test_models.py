"""Checks on the model code that need no model download. Run with: make test"""
import numpy as np
import pytest
import torch

from evaluation.evaluate import set_scores
from src.common import f1_counts, load_split, technique_pool, top_k_mask, tune_thresholds
from src.holdout import load_hidden
from src.retriever import doc_folds, multi_positive_loss


def test_loss_ignores_other_true_labels():
    # Sentence with gold A and B. B scoring very high must not hurt A's term.
    tau = 0.1
    sims = torch.tensor([[0.5, 0.9, 0.1, 0.2]])  # A, B, C, D
    positives = torch.tensor([[True, True, False, False]])
    loss = multi_positive_loss(sims, positives, tau)

    def term(p):
        negatives = torch.logsumexp(sims[0, 2:] / tau, 0)
        return torch.logaddexp(sims[0, p] / tau, negatives) - sims[0, p] / tau

    assert loss.item() == pytest.approx(((term(0) + term(1)) / 2).item(), rel=1e-5)


def test_loss_falls_when_positive_rises():
    positives = torch.tensor([[True, False, False]])
    low = multi_positive_loss(torch.tensor([[0.1, 0.5, 0.5]]), positives, 0.05)
    high = multi_positive_loss(torch.tensor([[0.9, 0.5, 0.5]]), positives, 0.05)
    assert high < low


def test_withheld_pool_and_labels():
    hidden = load_hidden()
    assert len(technique_pool("full", "train")) == 40
    train_pool = technique_pool("withheld", "train")
    assert len(train_pool) == 30 and not hidden & set(train_pool)
    assert not hidden & set(technique_pool("withheld", "dev"))
    assert hidden <= set(technique_pool("withheld", "test"))
    for split in ("train", "dev"):
        for r in load_split(split, "withheld"):
            assert not hidden & set(r["labels"]) and not hidden & set(r["gold"])


def test_test_split_is_never_filtered():
    assert len(load_split("test", "withheld")) == len(load_split("test", "full"))


def test_f1_counts_matches_evaluate():
    rng = np.random.default_rng(0)
    P, G = rng.random((50, 4)) < 0.3, rng.random((50, 4)) < 0.3
    labels = list("ABCD")
    keys = list(range(50))
    as_sets = lambda M: {i: {labels[j] for j in np.flatnonzero(M[i])} for i in keys}
    ref = set_scores(as_sets(G), as_sets(P), labels)
    mine = f1_counts(P, G)
    assert mine["micro_f1"] == pytest.approx(ref["micro"]["f1"])
    assert mine["macro_f1"] == pytest.approx(ref["macro"]["f1"])


def test_top_k_and_tuning():
    S = np.array([[0.9, 0.8, 0.1], [0.2, 0.7, 0.6]])
    G = np.array([[True, False, False], [False, True, False]])
    assert top_k_mask(S, 1).tolist() == [[True, False, False], [False, True, False]]
    best = tune_thresholds(S, G, ks=(1, 2))
    assert best["micro_f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------- mining and verifier

from src.mine_negatives import mine
from src.verifier import pipeline_matrix

ROWS = {("d", 1): {"labels": ["A"]}, ("d", 2): {"labels": []}}
CANDS = {("d", 1): [("A", 0.9), ("B", 0.95), ("C", 0.2)], ("d", 2): [("B", 0.5), ("C", 0.4), ("A", 0.1)]}


def test_mining_guard_and_sources():
    pairs, skipped, summary = mine(ROWS, CANDS, neg_k=10, unlabeled_k=2, unlabeled_frac=1.0, guard_quantile=0.5, seed=0)
    # Guard = the only gold score, 0.9. B (0.95) on the labeled sentence is skipped as a possible missed label.
    assert [(p["technique"], p["label"], p["source"]) for p in pairs] == [
        ("A", 1, "gold"), ("C", 0, "labeled_neg"), ("B", 0, "unlabeled_neg"), ("C", 0, "unlabeled_neg")]
    assert [p["technique"] for p in skipped] == ["B"]
    assert summary["guard"] == pytest.approx(0.9)


def test_pipeline_keeps_only_top_k_candidates():
    rows = [{"doc_id": "d", "sent_idx": 1, "labels": ["A"]}]
    verified = {("d", 1): [("B", 0.2), ("A", 0.9), ("C", 0.8)]}
    S, G = pipeline_matrix(rows, verified, ["A", "B", "C"], k=2)
    assert S[0].tolist() == [0.9, 0.2, -np.inf] and G[0].tolist() == [True, False, False]


def test_verifier_input_keeps_technique_text_whole():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT", local_files_only=True)
    except OSError:
        pytest.skip("SecureBERT tokenizer not downloaded")
    from src.common import in_scope_ids, technique_texts
    from src.verifier import PairBuilder
    builder = PairBuilder(tokenizer, in_scope_ids())
    long_sentence = "word " * 400
    for t, text in technique_texts(in_scope_ids()).items():
        full = tokenizer(text, add_special_tokens=False)["input_ids"]
        assert builder.tech[t] == full, f"{t} description was cut"
        ids = builder.ids(long_sentence, t)
        assert len(ids) <= 512 and ids[-len(full) - 1:-1] == full


def test_crossfit_folds_split_whole_documents():
    rows = load_split("train", "full")
    folds = doc_folds(rows, 5, seed=0)
    assert set(folds) == {r["doc_id"] for r in rows}  # every document gets exactly one fold
    sizes = np.bincount(list(folds.values()), minlength=5)
    assert sizes.max() - sizes.min() <= 1  # round-robin keeps document counts even
    assert folds == doc_folds(rows, 5, seed=0)  # same seed, same folds


def _history(dev_metric, train_loss, dev_loss):
    return [{"epoch": i + 1, "micro_f1": m, "loss": tl, "dev_loss": dl, "train_micro_f1": m + 0.05}
            for i, (m, tl, dl) in enumerate(zip(dev_metric, train_loss, dev_loss))]


def test_curves_verdicts():
    from evaluation.curves import diagnose
    falling = [0.5, 0.4, 0.3, 0.2, 0.1]
    # Dev score still climbing in the last allowed epoch.
    v = diagnose(_history([0.3, 0.4, 0.5, 0.55, 0.6], falling, [0.5, 0.45, 0.4, 0.38, 0.36]),
                 {"epochs": 5}, "micro_f1")
    assert v["verdict"].startswith("undertrained")
    # Dev loss bottoms at epoch 2 and climbs while train loss keeps falling.
    v = diagnose(_history([0.5, 0.6, 0.6, 0.59, 0.58], falling, [0.4, 0.3, 0.35, 0.4, 0.45]),
                 {"epochs": 10}, "micro_f1")
    assert v["verdict"].startswith("overfits after about epoch 2") and v["best_epoch"] == 2
    assert abs(v["gap_at_best"] - 0.05) < 1e-9
    # Score flat, dev loss flat: a plateau, not overfitting.
    v = diagnose(_history([0.5, 0.6, 0.6, 0.6, 0.6], falling, [0.4, 0.3, 0.3, 0.3, 0.3]),
                 {"epochs": 10}, "micro_f1")
    assert v["verdict"].startswith("plateau")
