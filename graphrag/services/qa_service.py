"""Q&A over local / hybrid pack-context or global community map-reduce."""

from __future__ import annotations

import asyncio

from graphrag.adapters.llm.base import LLMClient
from graphrag.api.schemas import QARequest, QAResponse, QASource, SearchMode, SearchResponse
from graphrag.config import Settings
from graphrag.exceptions import UpstreamModelError
from graphrag.services.retrieval_service import RetrievalService, pack_context

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided knowledge-graph context.
If the context is insufficient, say you do not know.
Cite relevant node names when helpful. Do not invent facts outside the context."""

MAP_SYSTEM = """You answer a question using ONLY one community summary from a knowledge graph.
Write a short partial answer (2-5 sentences). If the summary is irrelevant, reply exactly: NO_INFO."""

REDUCE_SYSTEM = """You synthesize a final answer from partial community answers.
Use only the partial answers. Resolve conflicts conservatively. If all say NO_INFO, say you do not know.
Be concise and cite community labels when helpful."""

EMPTY_CONTEXT_ANSWER = (
    "I do not know. The knowledge graph returned no matching context for this question."
)
GENERATION_FAILED_ANSWER = (
    "I could not generate an answer because the language model is unavailable. "
    "Sources from retrieval are listed below when available."
)


class QAService:
    def __init__(
        self,
        retrieval: RetrievalService,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.settings = settings

    async def ask(self, request: QARequest) -> QAResponse:
        mode_used = await self.retrieval.resolve_mode(request.question, request.mode)
        if mode_used == "global":
            return await self._ask_global(request, mode_used=mode_used)
        return await self._ask_pack(request, mode_used=mode_used)

    async def _ask_pack(self, request: QARequest, *, mode_used: SearchMode) -> QAResponse:
        search: SearchResponse = await self.retrieval.search(
            request.question,
            mode=mode_used,
            top_k=request.top_k,
            node_types=request.node_types,
            expand_hops=request.expand_hops,
            edge_types=request.edge_types,
        )
        return await self._answer_from_search(request, search)

    async def _answer_from_search(
        self, request: QARequest, search: SearchResponse
    ) -> QAResponse:
        mode_used = search.mode_used
        if not search.hits:
            return QAResponse(
                answer=EMPTY_CONTEXT_ANSWER,
                sources=[],
                subgraph=search.subgraph if request.include_sources else None,
                mode_used=mode_used,
                confidence=None,
                generation_error=None,
            )

        context = pack_context(search.hits, self.settings.context_token_budget)
        user_prompt = (
            f"Context:\n{context or '(no matching context)'}\n\n"
            f"Question: {request.question}\n\n"
            "Answer:"
        )
        try:
            answer = await self.llm.complete(system=SYSTEM_PROMPT, user=user_prompt)
        except UpstreamModelError as exc:
            return self._partial_response(request, search, error=str(exc))

        return QAResponse(
            answer=answer,
            sources=self._sources(request, search),
            subgraph=search.subgraph if request.include_sources else None,
            mode_used=mode_used,
            confidence=self._confidence(search),
            generation_error=None,
        )

    async def _ask_global(self, request: QARequest, *, mode_used: SearchMode) -> QAResponse:
        search = await self.retrieval.global_search(
            request.question,
            top_k=request.top_k or self.settings.global_map_top_k,
        )
        # Reuse hybrid fallback hits instead of re-searching via _ask_pack.
        if search.mode_used != "global":
            return await self._answer_from_search(request, search)

        sem = asyncio.Semaphore(self.settings.global_map_concurrency)

        async def _map_one(hit) -> tuple[str, str]:
            async with sem:
                user = (
                    f"Community: {hit.node_name}\n"
                    f"Summary:\n{hit.text}\n\n"
                    f"Question: {request.question}\n\n"
                    "Partial answer:"
                )
                partial = await self.llm.complete(system=MAP_SYSTEM, user=user, temperature=0.1)
                return hit.node_name, partial.strip()

        mapped = await asyncio.gather(
            *[_map_one(h) for h in search.hits],
            return_exceptions=True,
        )
        usable: list[tuple[str, str]] = []
        map_errors: list[str] = []
        for item in mapped:
            if isinstance(item, BaseException):
                map_errors.append(str(item))
                continue
            label, text = item
            if not text or text.upper() == "NO_INFO":
                continue
            usable.append((label, text))

        if not usable:
            err = (
                map_errors[0]
                if map_errors
                else "all community maps returned NO_INFO or failed"
            )
            return self._partial_response(request, search, error=err)

        partial_blocks = "\n\n".join(f"[{label}]\n{text}" for label, text in usable)
        reduce_user = (
            f"Question: {request.question}\n\n"
            f"Partial answers:\n{partial_blocks}\n\n"
            "Final answer:"
        )
        try:
            answer = await self.llm.complete(
                system=REDUCE_SYSTEM, user=reduce_user, temperature=0.2
            )
        except UpstreamModelError as exc:
            return self._partial_response(request, search, error=str(exc))

        return QAResponse(
            answer=answer,
            sources=self._sources(request, search),
            subgraph=search.subgraph if request.include_sources else None,
            mode_used=mode_used,
            confidence=self._confidence(search),
            generation_error=None,
        )

    def _partial_response(
        self, request: QARequest, search: SearchResponse, *, error: str
    ) -> QAResponse:
        return QAResponse(
            answer=GENERATION_FAILED_ANSWER,
            sources=self._sources(request, search),
            subgraph=search.subgraph if request.include_sources else None,
            mode_used=search.mode_used,
            confidence=self._confidence(search),
            generation_error=error,
        )

    def _sources(self, request: QARequest, search: SearchResponse) -> list[QASource]:
        if not request.include_sources:
            return []
        limit = request.top_k or self.settings.retrieval_top_k
        return [
            QASource(
                chunk_id=hit.chunk_id,
                node_id=hit.node_id,
                node_name=hit.node_name,
                excerpt=hit.text[:280],
                score=float(hit.score),
            )
            for hit in search.hits[:limit]
        ]

    def _confidence(self, search: SearchResponse) -> float | None:
        if not search.hits:
            return None
        limit = min(len(search.hits), self.settings.retrieval_top_k)
        scores = [max(0.0, min(1.0, float(h.score))) for h in search.hits[:limit]]
        if not scores:
            return None
        # Emphasize the best hit while still reflecting breadth.
        best = max(scores)
        mean = sum(scores) / len(scores)
        return round(0.65 * best + 0.35 * mean, 3)
