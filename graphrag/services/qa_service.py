"""Q&A over local / hybrid pack-context or global community map-reduce."""

from __future__ import annotations

import asyncio

from graphrag.adapters.llm.base import LLMClient
from graphrag.api.schemas import QARequest, QAResponse, QASource, SearchMode, SearchResponse
from graphrag.config import Settings
from graphrag.services.retrieval_service import RetrievalService, pack_context

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided knowledge-graph context.
If the context is insufficient, say you do not know.
Cite relevant node names when helpful. Do not invent facts outside the context."""

MAP_SYSTEM = """You answer a question using ONLY one community summary from a knowledge graph.
Write a short partial answer (2-5 sentences). If the summary is irrelevant, reply exactly: NO_INFO."""

REDUCE_SYSTEM = """You synthesize a final answer from partial community answers.
Use only the partial answers. Resolve conflicts conservatively. If all say NO_INFO, say you do not know.
Be concise and cite community labels when helpful."""


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
        mode_used = search.mode_used
        context = pack_context(search.hits, self.settings.context_token_budget)
        user_prompt = (
            f"Context:\n{context or '(no matching context)'}\n\n"
            f"Question: {request.question}\n\n"
            "Answer:"
        )
        answer = await self.llm.complete(system=SYSTEM_PROMPT, user=user_prompt)
        return QAResponse(
            answer=answer,
            sources=self._sources(request, search),
            subgraph=search.subgraph if request.include_sources else None,
            mode_used=mode_used,
        )

    async def _ask_global(self, request: QARequest, *, mode_used: SearchMode) -> QAResponse:
        search = await self.retrieval.global_search(
            request.question,
            top_k=request.top_k or self.settings.global_map_top_k,
        )
        # If global fell back to hybrid (no communities), use pack path.
        if search.mode_used != "global":
            return await self._ask_pack(
                QARequest(
                    question=request.question,
                    mode="hybrid",
                    top_k=request.top_k,
                    expand_hops=request.expand_hops,
                    node_types=request.node_types,
                    edge_types=request.edge_types,
                    include_sources=request.include_sources,
                ),
                mode_used="hybrid",
            )

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

        mapped = await asyncio.gather(*[_map_one(h) for h in search.hits])
        partial_blocks = "\n\n".join(
            f"[{label}]\n{text}" for label, text in mapped if text
        )
        reduce_user = (
            f"Question: {request.question}\n\n"
            f"Partial answers:\n{partial_blocks or '(none)'}\n\n"
            "Final answer:"
        )
        answer = await self.llm.complete(system=REDUCE_SYSTEM, user=reduce_user, temperature=0.2)
        return QAResponse(
            answer=answer,
            sources=self._sources(request, search),
            subgraph=search.subgraph if request.include_sources else None,
            mode_used=mode_used,
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
            )
            for hit in search.hits[:limit]
        ]
