"""Helpers for fixed-dimension embedding vectors."""

from __future__ import annotations

import math

from graphrag.exceptions import EmbeddingDimensionError


def fit_embedding(vec: list[float], dim: int) -> list[float]:
    """Truncate oversized vectors to ``dim`` (L2-renormalize); reject undersized ones."""
    n = len(vec)
    if n == dim:
        return vec
    if n < dim:
        raise EmbeddingDimensionError(
            f"Embedding dimension mismatch: got {n}, expected {dim}"
        )
    truncated = vec[:dim]
    norm = math.sqrt(sum(x * x for x in truncated))
    if norm == 0.0:
        return truncated
    return [x / norm for x in truncated]
