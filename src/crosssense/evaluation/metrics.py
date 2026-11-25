"""Hand-checkable paired retrieval metrics and query-bootstrap uncertainty."""

from collections.abc import Sequence

import numpy as np


def paired_ranks(scores: np.ndarray, query_ids: Sequence[str],
                 gallery_ids: Sequence[str]) -> np.ndarray:
    """Rank each query's exact paired ID; equal scores use gallery input order."""
    array = np.asarray(scores)
    if array.ndim != 2 or array.shape != (len(query_ids), len(gallery_ids)):
        raise ValueError("Score shape must match query and gallery IDs")
    if not len(query_ids) or not len(gallery_ids) or not np.isfinite(array).all():
        raise ValueError("Metrics require nonempty IDs and finite scores")
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("Query and gallery IDs must be unique")
    lookup = {key: i for i, key in enumerate(gallery_ids)}
    if any(key not in lookup for key in query_ids):
        raise ValueError("Every query requires a labelled positive in the gallery")
    order = np.argsort(-array, axis=1, kind="stable")
    return np.array([int(np.flatnonzero(order[i] == lookup[key])[0]) + 1
                     for i, key in enumerate(query_ids)], dtype=np.int64)


def metrics_from_ranks(ranks: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Recall@K is fraction with the single manifest positive among first K."""
    values = np.asarray(ranks, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 1):
        raise ValueError("Ranks must be a nonempty finite positive vector")
    if any(k < 1 for k in ks):
        raise ValueError("Recall cutoffs must be positive")
    return {**{f"recall@{k}": float(np.mean(values <= k)) for k in ks},
            "mrr": float(np.mean(1 / values))}


def retrieval_metrics(scores: np.ndarray, query_ids: Sequence[str],
                      gallery_ids: Sequence[str]) -> dict[str, float]:
    """Compute exact-identity Recall@1/5/10 and MRR."""
    return metrics_from_ranks(paired_ranks(scores, query_ids, gallery_ids))


def bootstrap_intervals(ranks: np.ndarray, seed: int = 42,
                        repetitions: int = 200) -> dict[str, list[float]]:
    """Return 95% percentile intervals from paired query resampling.

    These intervals condition on this gallery and are not independent spatial
    region or repeated-training-seed uncertainty estimates.
    """
    metrics_from_ranks(ranks)
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required")
    rng = np.random.default_rng(seed)
    samples = [metrics_from_ranks(rng.choice(ranks, len(ranks), replace=True))
               for _ in range(repetitions)]
    return {key: np.quantile([row[key] for row in samples], [0.025, 0.975]).tolist()
            for key in samples[0]}
