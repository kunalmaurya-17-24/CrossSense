import numpy as np
import pytest

from crosssense.evaluation.metrics import bootstrap_intervals, metrics_from_ranks, retrieval_metrics


def test_recall_and_mrr_hand_checkable():
    # a ranks first; b ranks second; c ranks third.
    scores = np.array([[3, 2, 1], [3, 2, 1], [3, 2, 1]], dtype=float)
    metrics = retrieval_metrics(scores, ["a", "b", "c"], ["a", "b", "c"])
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@5"] == metrics["recall@10"] == 1
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


def test_perfect_metrics_and_bootstrap_determinism():
    assert retrieval_metrics(np.eye(3), ["a", "b", "c"], ["a", "b", "c"])["mrr"] == 1
    assert bootstrap_intervals(np.array([1, 2, 3]), 42) == bootstrap_intervals(np.array([1, 2, 3]), 42)


def test_invalid_labels_and_empty_metrics():
    with pytest.raises(ValueError, match="nonempty"):
        metrics_from_ranks(np.array([]))
    with pytest.raises(ValueError, match="positive"):
        retrieval_metrics(np.eye(2), ["a", "c"], ["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        retrieval_metrics(np.eye(2), ["a", "a"], ["a", "b"])
