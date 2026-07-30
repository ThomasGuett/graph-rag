from openai import AsyncOpenAI

from graphrag.config import Settings


class OpenAICompatibleEmbeddings:
    """Embeddings via any OpenAI-compatible HTTP API."""

    def __init__(self, settings: Settings) -> None:
        self.dim = settings.embedding_dim
        self._model = settings.embedding_model
        self._client = AsyncOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self._model, input=texts)
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        for vec in vectors:
            if len(vec) != self.dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {len(vec)}, expected {self.dim}. "
                    f"Configure EMBEDDING_MODEL / EMBEDDING_DIM to match."
                )
        return vectors
