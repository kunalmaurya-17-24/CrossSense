import numpy as np
import pytest

from crosssense.retrieval.index import CosineIndex
from crosssense.retrieval.search import statistical_features


def index():
    return CosineIndex(np.array([[1, 0], [0, 2], [-1, 0]], dtype=float), ["a", "b", "c"],
                       [{"pair_id": key, "x": i * 1000} for i, key in enumerate(["a", "b", "c"])])


def test_exact_topk_correct_scores_metadata_and_size():
    gallery = index()
    result = gallery.search(np.array([3., 0]), top_k=99)[0]
    assert [row["id"] for row in result] == ["a", "b", "c"]
    np.testing.assert_allclose([row["score"] for row in result], [1, 0, -1])
    assert result[1]["metadata"]["x"] == 1000
    assert gallery.size_bytes > 3 * 2 * 4


def test_stable_ties_and_multiple_queries():
    result = index().search(np.array([[1, 1], [1, 1]], dtype=float), 2)
    assert [[row["id"] for row in hits] for hits in result] == [["a", "b"], ["a", "b"]]


def test_empty_duplicate_missing_metadata_gallery():
    with pytest.raises(ValueError, match="empty"):
        CosineIndex(np.empty((0, 2)), [], [])
    with pytest.raises(ValueError, match="unique"):
        CosineIndex(np.eye(2), ["a", "a"], [{"pair_id": "a"}] * 2)
    with pytest.raises(ValueError, match="metadata"):
        CosineIndex(np.eye(2), ["a", "b"], [{}, {}])
    with pytest.raises(ValueError, match="pair_id"):
        CosineIndex(np.eye(2), ["a", "b"], [{"pair_id": "b"}, {"pair_id": "a"}])


@pytest.mark.parametrize("query", [np.zeros(2), [np.nan, 1], [np.inf, 0], [1, 2, 3], np.empty((0, 2))])
def test_invalid_query(query):
    with pytest.raises(ValueError, match="Query|Queries"):
        index().search(query)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_topk(k):
    with pytest.raises(ValueError, match="top_k"):
        index().search(np.array([1, 0]), k)


def test_statistical_baseline_common_dimensions_and_finite():
    sar = statistical_features(np.ones((3, 2, 8, 8), dtype=np.float32))
    optical = statistical_features(np.ones((3, 4, 8, 8), dtype=np.float32))
    assert sar.shape == optical.shape == (3, 8)
    np.testing.assert_allclose(np.linalg.norm(sar, axis=1), 1, atol=1e-6)
    assert np.isfinite(statistical_features(np.zeros((2, 8, 8)))).all()
