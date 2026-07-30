"""In-process ingest job runner (no Celery)."""

from __future__ import annotations

import logging
from uuid import UUID

from graphrag.adapters.db.session import get_session_factory
from graphrag.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddings
from graphrag.adapters.llm.openai_compatible import OpenAICompatibleLLM
from graphrag.config import get_settings
from graphrag.services.embedding_service import EmbeddingService
from graphrag.services.ingest_service import IngestService

logger = logging.getLogger(__name__)


async def run_ingest_job(job_id: UUID) -> None:
    """Execute an ingest job in its own DB session."""
    settings = get_settings()
    factory = get_session_factory()
    embeddings = EmbeddingService(OpenAICompatibleEmbeddings(settings), settings)
    llm = OpenAICompatibleLLM(settings)

    async with factory() as session:
        service = IngestService(session, embeddings, llm, settings)
        try:
            await service.run_job(job_id)
            await session.commit()
        except Exception:
            # run_job already stamped failed status; commit that, don't roll it back.
            logger.exception("ingest job %s failed", job_id)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                logger.exception("failed to persist ingest job %s error state", job_id)
