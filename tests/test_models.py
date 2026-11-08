"""Checks on the model code that need no model download. Run with: make test"""
import numpy as np
import pytest
import torch

from evaluation.evaluate import set_scores
from src.common import f1_counts, load_split, technique_pool, top_k_mask, tune_thresholds
from src.holdout import load_hidden
from src.retriever import multi_positive_loss


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
