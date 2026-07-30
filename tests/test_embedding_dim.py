from graphrag.adapters.embeddings.dim import fit_embedding
from graphrag.exceptions import EmbeddingDimensionError
import pytest


def test_fit_embedding_passthrough():
    vec = [0.0, 1.0, 0.0]
    assert fit_embedding(vec, 3) == vec


def test_fit_embedding_truncates_and_renormalizes():
    vec = [3.0, 4.0, 0.0, 9.0]
    out = fit_embedding(vec, 2)
    assert len(out) == 2
    assert abs(out[0] - 0.6) < 1e-9
    assert abs(out[1] - 0.8) < 1e-9


def test_fit_embedding_rejects_short():
    with pytest.raises(EmbeddingDimensionError):
        fit_embedding([1.0, 2.0], 3)
