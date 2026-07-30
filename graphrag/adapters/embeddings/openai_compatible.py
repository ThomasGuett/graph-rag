from openai import AsyncOpenAI

from graphrag.adapters.embeddings.dim import fit_embedding
from graphrag.adapters.retry import with_retries
from graphrag.config import Settings


class OpenAICompatibleEmbeddings:
    """Embeddings via any OpenAI-compatible HTTP API."""

    def __init__(self, settings: Settings) -> None:
        self.dim = settings.embedding_dim
        self._model = settings.embedding_model
        self._base_url = settings.openai_api_base
        self._max_retries = settings.openai_max_retries
        self._client = AsyncOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=0,
        )

    async def embed(
        self, texts: list[str], *, max_retries: int | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        retries = self._max_retries if max_retries is None else max_retries

        async def _once() -> list[list[float]]:
            response = await self._client.embeddings.create(model=self._model, input=texts)
            vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
            return [fit_embedding(vec, self.dim) for vec in vectors]

        return await with_retries(
            _once,
            max_retries=retries,
            label=f"embeddings model={self._model} base={self._base_url}",
            detail_prefix="embedding request failed",
        )
