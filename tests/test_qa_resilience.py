"""Q&A degradation, empty-context, and global map resilience."""

from uuid import uuid4

import pytest

from graphrag.api.schemas import QARequest, SearchHit, SearchMode, SearchResponse, SubgraphOut
from graphrag.config import Settings
from graphrag.exceptions import UpstreamModelError
from graphrag.services.qa_service import (
    EMPTY_CONTEXT_ANSWER,
    GENERATION_FAILED_ANSWER,
    QAService,
)


class _RecordingLLM:
    def __init__(self, *, fail_on: set[str] | None = None, fail_always: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_on = fail_on or set()
        self.fail_always = fail_always
        self.map_count = 0

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls.append((system, user))
        if self.fail_always:
            raise UpstreamModelError("llm down")
        if "ONLY one community" in system:
            self.map_count += 1
            if "fail_map" in self.fail_on and self.map_count == 1:
                raise UpstreamModelError("map failed")
            label = "community"
            if "Community:" in user:
                label = user.split("Community:", 1)[1].split("\n", 1)[0].strip()
            if "NO_INFO" in self.fail_on and label.endswith("2"):
                return "NO_INFO"
            return f"partial about {label}"
        if "synthesize" in system.lower():
            if "fail_reduce" in self.fail_on:
                raise UpstreamModelError("reduce failed")
            return "final synthesized answer"
        if "fail_pack" in self.fail_on:
            raise UpstreamModelError("pack llm failed")
        return "single-shot answer"


def _hit(name: str = "Doc", score: float = 0.9, text: str = "Accounts are billed monthly.") -> SearchHit:
    return SearchHit(
        chunk_id=uuid4(),
        node_id=uuid4(),
        node_name=name,
        node_type="document",
        text=text,
        score=score,
        hop=0,
    )


class _Retrieval:
    def __init__(
        self,
        *,
        resolve: SearchMode = "hybrid",
        search_response: SearchResponse | None = None,
        global_response: SearchResponse | None = None,
    ) -> None:
        self._resolve = resolve
        self._search_response = search_response or SearchResponse(
            hits=[], subgraph=SubgraphOut(), mode_used="hybrid"
        )
        self._global_response = global_response
        self.search_calls = 0
        self.global_calls = 0

    async def resolve_mode(self, query: str, mode: SearchMode) -> SearchMode:
        return self._resolve if mode == "auto" else mode

    async def search(self, query: str, **kwargs):
        self.search_calls += 1
        return self._search_response

    async def global_search(self, query: str, *, top_k: int | None = None):
        self.global_calls += 1
        assert self._global_response is not None
        return self._global_response


@pytest.mark.asyncio
async def test_pack_empty_hits_skips_llm():
    llm = _RecordingLLM()
    retrieval = _Retrieval(
        resolve="hybrid",
        search_response=SearchResponse(hits=[], subgraph=SubgraphOut(), mode_used="hybrid"),
    )
    service = QAService(retrieval, llm, Settings(_env_file=None))  # type: ignore[arg-type]
    result = await service.ask(QARequest(question="Anything?", mode="hybrid"))
    assert result.answer == EMPTY_CONTEXT_ANSWER
    assert result.confidence is None
    assert result.generation_error is None
    assert llm.calls == []


@pytest.mark.asyncio
async def test_pack_llm_failure_returns_partial_with_sources():
    hit = _hit()
    llm = _RecordingLLM(fail_on={"fail_pack"})
    retrieval = _Retrieval(
        search_response=SearchResponse(
            hits=[hit], subgraph=SubgraphOut(), mode_used="hybrid"
        )
    )
    service = QAService(retrieval, llm, Settings(_env_file=None))  # type: ignore[arg-type]
    result = await service.ask(QARequest(question="When billed?", mode="hybrid"))
    assert result.answer == GENERATION_FAILED_ANSWER
    assert result.generation_error == "pack llm failed"
    assert len(result.sources) == 1
    assert result.sources[0].node_name == "Doc"
    assert result.confidence is not None
    assert result.mode_used == "hybrid"


@pytest.mark.asyncio
async def test_global_map_survives_one_map_failure():
    hits = [
        _hit("community_1", 0.9, "Summary A"),
        _hit("community_2", 0.8, "Summary B"),
    ]
    for h in hits:
        h.node_type = "community"
    llm = _RecordingLLM(fail_on={"fail_map"})
    retrieval = _Retrieval(
        resolve="global",
        global_response=SearchResponse(
            hits=hits, subgraph=SubgraphOut(), mode_used="global"
        ),
    )
    service = QAService(retrieval, llm, Settings(_env_file=None))  # type: ignore[arg-type]
    result = await service.ask(QARequest(question="Themes?", mode="global"))
    assert result.answer == "final synthesized answer"
    assert result.generation_error is None
    assert result.mode_used == "global"
    # one failed map + one success + reduce
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_global_reduce_failure_returns_partial():
    hits = [
        _hit("community_1", 0.9, "Summary A"),
        _hit("community_2", 0.8, "Summary B"),
    ]
    for h in hits:
        h.node_type = "community"
    llm = _RecordingLLM(fail_on={"fail_reduce"})
    retrieval = _Retrieval(
        resolve="global",
        global_response=SearchResponse(
            hits=hits, subgraph=SubgraphOut(), mode_used="global"
        ),
    )
    service = QAService(retrieval, llm, Settings(_env_file=None))  # type: ignore[arg-type]
    result = await service.ask(QARequest(question="Themes?", mode="global"))
    assert result.answer == GENERATION_FAILED_ANSWER
    assert result.generation_error == "reduce failed"
    assert len(result.sources) == 2


@pytest.mark.asyncio
async def test_global_hybrid_fallback_reuses_search_without_repack_search():
    hybrid_hit = _hit("FallbackDoc", 0.7, "Hybrid evidence.")
    llm = _RecordingLLM()
    retrieval = _Retrieval(
        resolve="global",
        global_response=SearchResponse(
            hits=[hybrid_hit], subgraph=SubgraphOut(), mode_used="hybrid"
        ),
        search_response=SearchResponse(
            hits=[_hit("ShouldNotUse")], subgraph=SubgraphOut(), mode_used="hybrid"
        ),
    )
    service = QAService(retrieval, llm, Settings(_env_file=None))  # type: ignore[arg-type]
    result = await service.ask(QARequest(question="Fallback?", mode="global"))
    assert result.answer == "single-shot answer"
    assert result.mode_used == "hybrid"
    assert result.sources[0].node_name == "FallbackDoc"
    assert retrieval.global_calls == 1
    assert retrieval.search_calls == 0


@pytest.mark.asyncio
async def test_confidence_emphasizes_best_hit():
    hits = [_hit("A", 1.0), _hit("B", 0.0)]
    llm = _RecordingLLM()
    retrieval = _Retrieval(
        search_response=SearchResponse(
            hits=hits, subgraph=SubgraphOut(), mode_used="hybrid"
        )
    )
    service = QAService(retrieval, llm, Settings(_env_file=None))  # type: ignore[arg-type]
    result = await service.ask(QARequest(question="?", mode="hybrid"))
    # 0.65 * 1.0 + 0.35 * 0.5 = 0.825
    assert result.confidence == 0.825
