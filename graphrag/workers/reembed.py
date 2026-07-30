"""Optional async workers for embed backfill / ingest."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Chunk
from graphrag.services.embedding_service import EmbeddingService


async def reembed_chunks_missing_vectors(
    session: AsyncSession,
    embeddings: EmbeddingService,
    *,
    limit: int = 100,
) -> list[UUID]:
    """
    Re-embed chunks that have NULL embeddings.

    Intended for admin/backfill jobs; the sync API embeds on create/update.
    """
    stmt = (
        select(Chunk)
        .where(Chunk.embedding.is_(None))
        .order_by(Chunk.created_at.asc())
        .limit(limit)
    )
    chunks = list((await session.execute(stmt)).scalars().all())
    if not chunks:
        return []
    vectors = await embeddings.embed_texts([c.text for c in chunks])
    updated: list[UUID] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector
        updated.append(chunk.id)
    await session.flush()
    return updated
