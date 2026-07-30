from graphrag.adapters.llm.base import LLMClient
from graphrag.api.schemas import QARequest, QAResponse, QASource, SearchResponse
from graphrag.config import Settings
from graphrag.services.retrieval_service import RetrievalService, pack_context

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided knowledge-graph context.
If the context is insufficient, say you do not know.
Cite relevant node names when helpful. Do not invent facts outside the context."""


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
        search: SearchResponse = await self.retrieval.hybrid_search(
            request.question,
            top_k=request.top_k,
            node_types=request.node_types,
            expand_hops=request.expand_hops,
            edge_types=request.edge_types,
        )
        context = pack_context(search.hits, self.settings.context_token_budget)
        user_prompt = (
            f"Context:\n{context or '(no matching context)'}\n\n"
            f"Question: {request.question}\n\n"
            "Answer:"
        )
        answer = await self.llm.complete(system=SYSTEM_PROMPT, user=user_prompt)

        sources: list[QASource] = []
        if request.include_sources:
            sources = [
                QASource(
                    chunk_id=hit.chunk_id,
                    node_id=hit.node_id,
                    node_name=hit.node_name,
                    excerpt=hit.text[:280],
                )
                for hit in search.hits[: request.top_k or self.settings.retrieval_top_k]
            ]

        return QAResponse(
            answer=answer,
            sources=sources,
            subgraph=search.subgraph if request.include_sources else None,
        )
