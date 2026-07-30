from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from graphrag.api.deps import get_edge_service
from graphrag.api.schemas import EdgeCreate, EdgeOut, EdgeUpdate
from graphrag.services.edge_service import EdgeService

router = APIRouter(prefix="/edges", tags=["edges"])


@router.post("", response_model=EdgeOut, status_code=status.HTTP_201_CREATED)
async def create_edge(
    body: EdgeCreate,
    service: EdgeService = Depends(get_edge_service),
) -> EdgeOut:
    edge = await service.create(body)
    return EdgeOut.model_validate(edge)


@router.get("", response_model=list[EdgeOut])
async def list_edges(
    src_id: UUID | None = None,
    dst_id: UUID | None = None,
    type: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: EdgeService = Depends(get_edge_service),
) -> list[EdgeOut]:
    edges = await service.list(src_id=src_id, dst_id=dst_id, type=type, limit=limit, offset=offset)
    return [EdgeOut.model_validate(e) for e in edges]


@router.get("/{edge_id}", response_model=EdgeOut)
async def get_edge(
    edge_id: UUID,
    service: EdgeService = Depends(get_edge_service),
) -> EdgeOut:
    edge = await service.get(edge_id)
    if not edge:
        raise HTTPException(status_code=404, detail="edge not found")
    return EdgeOut.model_validate(edge)


@router.patch("/{edge_id}", response_model=EdgeOut)
async def update_edge(
    edge_id: UUID,
    body: EdgeUpdate,
    service: EdgeService = Depends(get_edge_service),
) -> EdgeOut:
    edge = await service.update(edge_id, body)
    if not edge:
        raise HTTPException(status_code=404, detail="edge not found")
    return EdgeOut.model_validate(edge)


@router.delete("/{edge_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_edge(
    edge_id: UUID,
    service: EdgeService = Depends(get_edge_service),
) -> None:
    deleted = await service.delete(edge_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="edge not found")
