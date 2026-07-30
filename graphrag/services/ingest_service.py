"""Document ingest orchestration: chunk → extract → resolve → communities."""

from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Chunk, Document, Edge, IngestJob, Node
from graphrag.adapters.llm.base import LLMClient
from graphrag.api.schemas import ChunkCreate
from graphrag.config import Settings
from graphrag.exceptions import NotFoundError
from graphrag.services.chunk_service import ChunkService
from graphrag.services.chunking_service import chunk_text
from graphrag.services.community_service import CommunityService
from graphrag.services.embedding_service import EmbeddingService
from graphrag.services.entity_resolution_service import EntityResolutionService
from graphrag.services.extraction_service import ExtractionResult, ExtractionService
from graphrag.services.graph_builder_service import GraphBuilderService


class IngestService:
    def __init__(
        self,
        session: AsyncSession,
        embeddings: EmbeddingService,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self.session = session
        self.embeddings = embeddings
        self.llm = llm
        self.settings = settings
        self.chunks = ChunkService(session, embeddings)
        self.extraction = ExtractionService(llm)
        self.communities = CommunityService(session, llm, embeddings, settings)

    async def create_document(
        self,
        *,
        title: str,
        text: str,
        source_uri: str | None = None,
        props: dict | None = None,
    ) -> tuple[Document, IngestJob]:
        doc_node = Node(
            type="document",
            name=title.strip(),
            props={"source_uri": source_uri, **(props or {})},
        )
        self.session.add(doc_node)
        await self.session.flush()

        document = Document(
            title=title.strip(),
            text=text,
            source_uri=source_uri,
            status="pending",
            node_id=doc_node.id,
            props=props or {},
        )
        self.session.add(document)
        await self.session.flush()

        job = IngestJob(
            document_id=document.id,
            stage="pending",
            status="pending",
            progress={},
        )
        self.session.add(job)
        await self.session.flush()
        await self.session.refresh(document)
        await self.session.refresh(job)
        return document, job

    async def get_document(self, document_id: UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def list_documents(self, *, limit: int = 50, offset: int = 0) -> list[Document]:
        stmt = (
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def document_counts(self, document: Document) -> dict[str, int]:
        chunk_count = 0
        mention_count = 0
        if document.node_id:
            chunk_count = int(
                (
                    await self.session.execute(
                        select(func.count()).select_from(Chunk).where(Chunk.node_id == document.node_id)
                    )
                ).scalar_one()
            )
            mention_count = int(
                (
                    await self.session.execute(
                        select(func.count())
                        .select_from(Edge)
                        .where(Edge.src_id == document.node_id, Edge.type == "mentions")
                    )
                ).scalar_one()
            )
        return {"chunks": chunk_count, "mentions": mention_count}

    async def get_job(self, job_id: UUID) -> IngestJob | None:
        return await self.session.get(IngestJob, job_id)

    async def start_reindex(self, document_id: UUID) -> IngestJob:
        document = await self.get_document(document_id)
        if not document:
            raise NotFoundError("document not found")
        await self._reset_document_derived(document)
        document.status = "pending"
        document.error = None
        job = IngestJob(
            document_id=document.id,
            stage="pending",
            status="pending",
            progress={},
        )
        self.session.add(job)
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def run_job(self, job_id: UUID) -> None:
        job = await self.get_job(job_id)
        if not job:
            raise NotFoundError("ingest job not found")
        document = await self.get_document(job.document_id)
        if not document:
            raise NotFoundError("document not found")

        job.status = "running"
        await self.session.flush()
        document_id = document.id
        current_job_id = job.id

        try:
            await self._set_stage(document, job, "chunking")
            chunk_ids = await self._chunk_and_embed(document)
            job.progress = {**(job.progress or {}), "chunks": len(chunk_ids)}
            await self.session.commit()

            document = await self.get_document(document_id)
            job = await self.get_job(current_job_id)
            assert document is not None and job is not None

            await self._set_stage(document, job, "extracting")
            extractions = await self._extract_chunks(chunk_ids)
            job.progress = {
                **(job.progress or {}),
                "extractions": len(extractions),
            }
            await self.session.commit()

            document = await self.get_document(document_id)
            job = await self.get_job(current_job_id)
            assert document is not None and job is not None

            await self._set_stage(document, job, "resolving")
            stats = await self._resolve_and_write(document, extractions)
            job.progress = {**(job.progress or {}), **stats}
            await self.session.commit()

            document = await self.get_document(document_id)
            job = await self.get_job(current_job_id)
            assert document is not None and job is not None

            await self._set_stage(document, job, "building_communities")
            communities = await self.communities.rebuild()
            job.progress = {
                **(job.progress or {}),
                "communities": len(communities),
            }
            document.status = "ready"
            document.error = None
            job.stage = "ready"
            job.status = "completed"
            await self.session.commit()
        except Exception as exc:
            await self.session.rollback()
            document = await self.get_document(document_id)
            job = await self.get_job(current_job_id)
            if document:
                document.status = "failed"
                document.error = str(exc)
            if job:
                job.status = "failed"
                job.error = str(exc)
            await self.session.commit()
            raise

    async def _set_stage(self, document: Document, job: IngestJob, stage: str) -> None:
        document.status = stage
        job.stage = stage
        await self.session.flush()

    async def _chunk_and_embed(self, document: Document) -> list[UUID]:
        if not document.node_id:
            raise NotFoundError("document node_id missing")
        # Remove prior document chunks before re-chunking.
        await self.session.execute(delete(Chunk).where(Chunk.node_id == document.node_id))
        await self.session.flush()

        spans = chunk_text(
            document.text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not spans:
            return []
        items = [
            ChunkCreate(node_id=document.node_id, text=span.text, props=span.props)
            for span in spans
        ]
        chunks = await self.chunks.create_batch(items)
        return [c.id for c in chunks]

    async def _extract_chunks(
        self, chunk_ids: list[UUID]
    ) -> list[tuple[UUID, ExtractionResult]]:
        if not chunk_ids:
            return []
        chunks = list(
            (await self.session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids))))
            .scalars()
            .all()
        )
        sem = asyncio.Semaphore(self.settings.extraction_concurrency)

        async def _one(chunk: Chunk) -> tuple[UUID, ExtractionResult]:
            async with sem:
                extraction = await self.extraction.extract_from_text(chunk.text)
                return chunk.id, extraction

        # Sequential-ish via semaphore; gather is OK because extraction is LLM-only (no ORM I/O).
        return list(await asyncio.gather(*[_one(c) for c in chunks]))

    async def _resolve_and_write(
        self,
        document: Document,
        extractions: list[tuple[UUID, ExtractionResult]],
    ) -> dict[str, int]:
        if not document.node_id:
            raise NotFoundError("document node_id missing")
        resolver = EntityResolutionService(self.session)
        builder = GraphBuilderService(self.session, resolver, self.embeddings)
        totals = {
            "entities": 0,
            "mentions": 0,
            "relationships": 0,
            "description_chunks": 0,
        }
        for chunk_id, extraction in extractions:
            stats = await builder.apply_extraction(
                document_node_id=document.node_id,
                chunk_id=chunk_id,
                extraction=extraction,
            )
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
        return totals

    async def _reset_document_derived(self, document: Document) -> None:
        """Clear document chunks and mentions edges before reindex."""
        if not document.node_id:
            return
        await self.session.execute(delete(Chunk).where(Chunk.node_id == document.node_id))
        await self.session.execute(
            delete(Edge).where(Edge.src_id == document.node_id, Edge.type == "mentions")
        )
        await self.session.flush()
