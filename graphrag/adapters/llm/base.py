from typing import Protocol


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> str:
        ...
