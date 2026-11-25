"""Exact cosine index returning scores and original geographic metadata."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class CosineIndex:
    """In-memory exact search; similarity scores are not probabilities."""

    def __init__(self, embeddings: np.ndarray, ids: Sequence[str],
                 metadata: Sequence[Mapping[str, Any]]) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0 or vectors.shape[1] == 0:
            raise ValueError("Retrieval gallery is empty or malformed; build a nonempty index first")
        if len(ids) != len(vectors) or len(metadata) != len(vectors):
            raise ValueError("Gallery embeddings, IDs and metadata must have equal lengths")
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Gallery IDs must be unique nonempty strings")
        if any(not isinstance(row, Mapping) or not row or "pair_id" not in row for row in metadata):
            raise ValueError("Each gallery item requires metadata including pair_id")
        if any(row["pair_id"] != key for row, key in zip(metadata, ids)):
            raise ValueError("Metadata pair_id must agree with its gallery ID")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
            raise ValueError("Gallery embeddings must be finite nonzero vectors")
        self.embeddings = np.ascontiguousarray(vectors / norms)
        self.ids = list(ids)
        self.metadata = [dict(row) for row in metadata]

    @property
    def size_bytes(self) -> int:
        """Report vector bytes plus UTF-8 serialized IDs and metadata (no Python overhead)."""
        return self.embeddings.nbytes + len(json.dumps(
            {"ids": self.ids, "metadata": self.metadata}, default=str).encode("utf-8"))

    def similarities(self, query: np.ndarray) -> np.ndarray:
        """Return the complete exact cosine score matrix for one or more queries."""
        vectors = np.asarray(query, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.ndim != 2 or len(vectors) == 0 or vectors.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query must be a nonempty vector or matrix with the index embedding dimension")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
            raise ValueError("Queries must contain finite nonzero embeddings")
        return (vectors / norms) @ self.embeddings.T

    def search(self, query: np.ndarray, top_k: int = 10) -> list[list[dict[str, Any]]]:
        """Return descending top-k IDs, cosine scores and pair/geographic metadata."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        scores = self.similarities(query)
        # Stable ordering resolves equal scores in original manifest order.
        order = np.argsort(-scores, axis=1, kind="stable")[:, :min(top_k, len(self.ids))]
        return [[{"id": self.ids[int(j)], "score": float(scores[i, j]),
                  "metadata": self.metadata[int(j)]} for j in rows]
                for i, rows in enumerate(order)]
