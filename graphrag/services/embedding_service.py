"""Embedding use-case wrapper around an EmbeddingClient."""

from __future__ import annotations

from graphrag.adapters.embeddings.base import EmbeddingClient
from graphrag.config import Settings
from graphrag.exceptions import EmbeddingDimensionError


class EmbeddingService:
    def __init__(self, client: EmbeddingClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self.dim = settings.embedding_dim

    async def embed_query(self, query: str) -> list[float]:
        vectors = await self.embed_texts([query])
        return vectors[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batch_size = max(1, self._settings.embedding_batch_size)
        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = await self._client.embed(batch)
            if len(vectors) != len(batch):
                raise EmbeddingDimensionError(
                    f"Embedding provider returned {len(vectors)} vectors for {len(batch)} texts"
                )
            for vec in vectors:
                if len(vec) != self.dim:
                    raise EmbeddingDimensionError(
                        f"Embedding dimension mismatch: got {len(vec)}, expected {self.dim}"
                    )
            out.extend(vectors)
        return out
