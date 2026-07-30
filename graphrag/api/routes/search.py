from fastapi import APIRouter, Depends

from graphrag.api.deps import get_retrieval_service
from graphrag.api.schemas import SearchRequest, SearchResponse
from graphrag.services.retrieval_service import RetrievalService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    return await service.hybrid_search(
        body.query,
        top_k=body.top_k,
        node_types=body.node_types,
        expand_hops=body.expand_hops,
        edge_types=body.edge_types,
    )
