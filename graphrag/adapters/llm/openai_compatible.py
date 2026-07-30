from openai import APIError, AsyncOpenAI, OpenAIError

from graphrag.config import Settings
from graphrag.exceptions import UpstreamModelError


class OpenAICompatibleLLM:
    """Chat completions via any OpenAI-compatible HTTP API."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model
        self._client = AsyncOpenAI(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
        )

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except (APIError, OpenAIError) as exc:
            raise UpstreamModelError(f"llm request failed: {exc}") from exc
        content = response.choices[0].message.content
        return content or ""
