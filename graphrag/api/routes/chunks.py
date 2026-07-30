from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from graphrag.adapters.db.models import Chunk
from graphrag.api.deps import get_chunk_service
from graphrag.api.schemas import ChunkBatchCreate, ChunkCreate, ChunkOut, ChunkUpdate
from graphrag.services.chunk_service import ChunkService

router = APIRouter(prefix="/chunks", tags=["chunks"])


def _chunk_out(chunk: Chunk, include_embedding: bool) -> ChunkOut:
    return ChunkOut(
        id=chunk.id,
        node_id=chunk.node_id,
        text=chunk.text,
        props=chunk.props or {},
        created_at=chunk.created_at,
        updated_at=chunk.updated_at,
        embedding=(
            list(chunk.embedding)
            if include_embedding and chunk.embedding is not None
            else None
        ),
    )


@router.post("", response_model=ChunkOut, status_code=status.HTTP_201_CREATED)
async def create_chunk(
    body: ChunkCreate,
    include_embedding: bool = False,
    service: ChunkService = Depends(get_chunk_service),
) -> ChunkOut:
    try:
        chunk = await service.create(body)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _chunk_out(chunk, include_embedding)


@router.post("/batch", response_model=list[ChunkOut], status_code=status.HTTP_201_CREATED)
async def create_chunks_batch(
    body: ChunkBatchCreate,
    include_embedding: bool = False,
    service: ChunkService = Depends(get_chunk_service),
) -> list[ChunkOut]:
    try:
        chunks = await service.create_batch(body.chunks)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [_chunk_out(c, include_embedding) for c in chunks]


@router.get("", response_model=list[ChunkOut])
async def list_chunks(
    node_id: UUID | None = None,
    include_embedding: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: ChunkService = Depends(get_chunk_service),
) -> list[ChunkOut]:
    chunks = await service.list(node_id=node_id, limit=limit, offset=offset)
    return [_chunk_out(c, include_embedding) for c in chunks]


@router.get("/{chunk_id}", response_model=ChunkOut)
async def get_chunk(
    chunk_id: UUID,
    include_embedding: bool = False,
    service: ChunkService = Depends(get_chunk_service),
) -> ChunkOut:
    chunk = await service.get(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="chunk not found")
    return _chunk_out(chunk, include_embedding)


@router.patch("/{chunk_id}", response_model=ChunkOut)
async def update_chunk(
    chunk_id: UUID,
    body: ChunkUpdate,
    include_embedding: bool = False,
    service: ChunkService = Depends(get_chunk_service),
) -> ChunkOut:
    try:
        chunk = await service.update(chunk_id, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not chunk:
        raise HTTPException(status_code=404, detail="chunk not found")
    return _chunk_out(chunk, include_embedding)


@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chunk(
    chunk_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
) -> None:
    deleted = await service.delete(chunk_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="chunk not found")
