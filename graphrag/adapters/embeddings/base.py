from typing import Protocol


class EmbeddingClient(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
