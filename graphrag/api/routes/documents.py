from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.api.deps import get_db_session, get_ingest_service
from graphrag.api.schemas import (
    DocumentCreate,
    DocumentCreateResponse,
    DocumentOut,
    IngestJobOut,
)
from graphrag.services.ingest_service import IngestService
from graphrag.workers.ingest import run_ingest_job

router = APIRouter(prefix="/documents", tags=["documents"])


def _document_out(document, *, counts: dict | None = None, job=None) -> DocumentOut:
    out = DocumentOut.model_validate(document)
    if counts is not None:
        out.counts = counts
    if job is not None:
        out.job = IngestJobOut.model_validate(job)
    return out


@router.post("", response_model=DocumentCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    body: DocumentCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
    service: IngestService = Depends(get_ingest_service),
) -> DocumentCreateResponse:
    document, job = await service.create_document(
        title=body.title,
        text=body.text,
        source_uri=body.source_uri,
        props=body.props,
    )
    # Commit before background task so the worker session can see the rows.
    await session.commit()
    background_tasks.add_task(run_ingest_job, job.id)
    return DocumentCreateResponse(
        document=_document_out(document, job=job),
        job=IngestJobOut.model_validate(job),
    )


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    limit: int = Query(default=50, ge=1, le=500),
    service: IngestService = Depends(get_ingest_service),
    offset: int = Query(default=0, ge=0),
) -> list[DocumentOut]:
    documents = await service.list_documents(limit=limit, offset=offset)
    out: list[DocumentOut] = []
    for document in documents:
        counts = await service.document_counts(document)
        out.append(_document_out(document, counts=counts))
    return out


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: UUID,
    service: IngestService = Depends(get_ingest_service),
) -> DocumentOut:
    document = await service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="document not found")
    counts = await service.document_counts(document)
    return _document_out(document, counts=counts)


@router.post("/{document_id}/reindex", response_model=DocumentCreateResponse)
async def reindex_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
    service: IngestService = Depends(get_ingest_service),
) -> DocumentCreateResponse:
    document = await service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="document not found")
    job = await service.start_reindex(document_id)
    document = await service.get_document(document_id)
    assert document is not None
    await session.commit()
    background_tasks.add_task(run_ingest_job, job.id)
    return DocumentCreateResponse(
        document=_document_out(document, job=job),
        job=IngestJobOut.model_validate(job),
    )
