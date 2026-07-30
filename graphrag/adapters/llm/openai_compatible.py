from openai import AsyncOpenAI

from graphrag.adapters.retry import with_retries
from graphrag.config import Settings


class OpenAICompatibleLLM:
    """Chat completions via any OpenAI-compatible HTTP API."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model
        self._base_url = settings.openai_api_base
        self._max_retries = settings.openai_max_retries
        self._client = AsyncOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=0,  # we handle retries ourselves for transient classification
        )

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_retries: int | None = None,
    ) -> str:
        retries = self._max_retries if max_retries is None else max_retries

        async def _once() -> str:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = response.choices[0].message.content
            return content or ""

        return await with_retries(
            _once,
            max_retries=retries,
            label=f"llm complete model={self._model} base={self._base_url}",
            detail_prefix="llm request failed",
        )
