from openai import APIError, AsyncOpenAI, OpenAIError

from graphrag.adapters.embeddings.dim import fit_embedding
from graphrag.config import Settings
from graphrag.exceptions import UpstreamModelError


class OpenAICompatibleEmbeddings:
    """Embeddings via any OpenAI-compatible HTTP API."""

    def __init__(self, settings: Settings) -> None:
        self.dim = settings.embedding_dim
        self._model = settings.embedding_model
        self._client = AsyncOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(model=self._model, input=texts)
        except (APIError, OpenAIError) as exc:
            raise UpstreamModelError(f"embedding request failed: {exc}") from exc
        vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        return [fit_embedding(vec, self.dim) for vec in vectors]
