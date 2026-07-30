import pytest

from graphrag.config import Settings
from graphrag.exceptions import EmbeddingDimensionError
from graphrag.services.embedding_service import EmbeddingService


class _FakeClient:
    dim = 2048

    def __init__(self, *, bad_dim: bool = False):
        self.bad_dim = bad_dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.bad_dim:
            return [[0.0] * 3 for _ in texts]
        return [[float(i)] * 2048 for i, _ in enumerate(texts)]


@pytest.mark.asyncio
async def test_embedding_service_batches_and_validates_dim():
    settings = Settings(embedding_dim=2048, embedding_batch_size=2, _env_file=None)
    client = _FakeClient()
    service = EmbeddingService(client, settings)
    out = await service.embed_texts(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == 2048 for v in out)
    assert client.calls == [["a", "b"], ["c"]]


@pytest.mark.asyncio
async def test_embedding_service_rejects_bad_dim():
    settings = Settings(embedding_dim=2048, embedding_batch_size=8, _env_file=None)
    client = _FakeClient(bad_dim=True)
    service = EmbeddingService(client, settings)
    with pytest.raises(EmbeddingDimensionError):
        await service.embed_texts(["x"])
