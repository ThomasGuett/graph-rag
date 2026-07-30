import pytest

from graphrag.api.schemas import QARequest, SearchHit, SearchMode, SearchResponse, SubgraphOut
from graphrag.config import Settings
from graphrag.services.qa_service import QAService


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls.append((system, user))
        if "ONLY one community" in system:
            label = "community"
            if "Community:" in user:
                label = user.split("Community:", 1)[1].split("\n", 1)[0].strip()
            return f"partial about {label}"
        if "synthesize" in system.lower():
            return "final synthesized answer"
        return "single-shot answer"


class _FakeRetrieval:
    def __init__(self, mode: SearchMode = "global") -> None:
        self._mode = mode

    async def resolve_mode(self, query: str, mode: SearchMode) -> SearchMode:
        return self._mode if mode == "auto" else mode

    async def search(self, query: str, **kwargs):
        return SearchResponse(
            hits=[],
            subgraph=SubgraphOut(),
            mode_used=kwargs.get("mode", "hybrid"),
        )

    async def global_search(self, query: str, *, top_k: int | None = None):
        from uuid import uuid4

        hits = [
            SearchHit(
                chunk_id=uuid4(),
                node_id=uuid4(),
                node_name="community_1",
                node_type="community",
                text="Summary about oncology.",
                score=0.9,
                hop=0,
            ),
            SearchHit(
                chunk_id=uuid4(),
                node_id=uuid4(),
                node_name="community_2",
                node_type="community",
                text="Summary about Boston.",
                score=0.8,
                hop=0,
            ),
        ]
        return SearchResponse(hits=hits, subgraph=SubgraphOut(), mode_used="global")


@pytest.mark.asyncio
async def test_qa_global_map_reduce_calls_map_then_reduce():
    llm = _FakeLLM()
    settings = Settings(_env_file=None)
    service = QAService(_FakeRetrieval("global"), llm, settings)  # type: ignore[arg-type]
    result = await service.ask(QARequest(question="What are the themes?", mode="global"))
    assert result.answer == "final synthesized answer"
    assert result.mode_used == "global"
    assert len(llm.calls) == 3
    assert "ONLY one community" in llm.calls[0][0]
    assert "ONLY one community" in llm.calls[1][0]
    assert "synthesize" in llm.calls[2][0].lower()


@pytest.mark.asyncio
async def test_qa_hybrid_single_shot():
    llm = _FakeLLM()
    settings = Settings(_env_file=None)

    class _HybridRetrieval(_FakeRetrieval):
        async def search(self, query: str, **kwargs):
            from uuid import uuid4

            hit = SearchHit(
                chunk_id=uuid4(),
                node_id=uuid4(),
                node_name="Doc",
                node_type="document",
                text="Accounts are billed monthly.",
                score=0.9,
                hop=0,
            )
            return SearchResponse(
                hits=[hit], subgraph=SubgraphOut(), mode_used="hybrid"
            )

    service = QAService(_HybridRetrieval("hybrid"), llm, settings)  # type: ignore[arg-type]
    result = await service.ask(QARequest(question="When billed?", mode="hybrid"))
    assert result.answer == "single-shot answer"
    assert result.mode_used == "hybrid"
    assert len(llm.calls) == 1
