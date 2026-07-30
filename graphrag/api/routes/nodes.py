from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from graphrag.api.deps import get_node_service
from graphrag.api.schemas import NodeCreate, NodeOut, NodeUpdate, SubgraphOut
from graphrag.services.node_service import NodeService

router = APIRouter(prefix="/nodes", tags=["nodes"])


@router.post("", response_model=NodeOut, status_code=status.HTTP_201_CREATED)
async def create_node(
    body: NodeCreate,
    service: NodeService = Depends(get_node_service),
) -> NodeOut:
    node = await service.create(body)
    return NodeOut.model_validate(node)


@router.get("", response_model=list[NodeOut])
async def list_nodes(
    type: str | None = None,
    q: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: NodeService = Depends(get_node_service),
) -> list[NodeOut]:
    nodes = await service.list(type=type, q=q, limit=limit, offset=offset)
    return [NodeOut.model_validate(n) for n in nodes]


@router.get("/{node_id}", response_model=NodeOut)
async def get_node(
    node_id: UUID,
    service: NodeService = Depends(get_node_service),
) -> NodeOut:
    node = await service.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="node not found")
    return NodeOut.model_validate(node)


@router.patch("/{node_id}", response_model=NodeOut)
async def update_node(
    node_id: UUID,
    body: NodeUpdate,
    service: NodeService = Depends(get_node_service),
) -> NodeOut:
    node = await service.update(node_id, body)
    if not node:
        raise HTTPException(status_code=404, detail="node not found")
    return NodeOut.model_validate(node)


@router.delete("/{node_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_node(
    node_id: UUID,
    service: NodeService = Depends(get_node_service),
) -> None:
    deleted = await service.delete(node_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="node not found")


@router.get("/{node_id}/neighbors", response_model=SubgraphOut)
async def get_neighbors(
    node_id: UUID,
    direction: str = Query(default="both", pattern="^(in|out|both)$"),
    type: str | None = None,
    depth: int = Query(default=1, ge=0, le=5),
    service: NodeService = Depends(get_node_service),
) -> SubgraphOut:
    node = await service.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="node not found")
    return await service.neighbors(node_id, direction=direction, edge_type=type, depth=depth)
