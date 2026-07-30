from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Chunk, Node
from graphrag.api.schemas import ChunkCreate, ChunkUpdate
from graphrag.exceptions import NotFoundError
from graphrag.services.embedding_service import EmbeddingService


class ChunkService:
    def __init__(self, session: AsyncSession, embeddings: EmbeddingService) -> None:
        self.session = session
        self.embeddings = embeddings

    async def create(self, data: ChunkCreate) -> Chunk:
        node = await self.session.get(Node, data.node_id)
        if not node:
            raise NotFoundError("node_id not found")
        vectors = await self.embeddings.embed_texts([data.text])
        chunk = Chunk(
            node_id=data.node_id,
            text=data.text,
            props=data.props or {},
            embedding=vectors[0],
        )
        self.session.add(chunk)
        await self.session.flush()
        await self.session.refresh(chunk)
        return chunk

    async def create_batch(self, items: list[ChunkCreate]) -> list[Chunk]:
        node_ids = {item.node_id for item in items}
        existing = list(
            (await self.session.execute(select(Node.id).where(Node.id.in_(node_ids)))).scalars().all()
        )
        if len(existing) != len(node_ids):
            raise NotFoundError("one or more node_id values not found")

        vectors = await self.embeddings.embed_texts([item.text for item in items])
        chunks: list[Chunk] = []
        for item, vector in zip(items, vectors, strict=True):
            chunk = Chunk(
                node_id=item.node_id,
                text=item.text,
                props=item.props or {},
                embedding=vector,
            )
            self.session.add(chunk)
            chunks.append(chunk)
        await self.session.flush()
        for chunk in chunks:
            await self.session.refresh(chunk)
        return chunks

    async def get(self, chunk_id: UUID) -> Chunk | None:
        return await self.session.get(Chunk, chunk_id)

    async def list(
        self,
        *,
        node_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Chunk]:
        stmt = select(Chunk).order_by(Chunk.created_at.desc()).limit(limit).offset(offset)
        if node_id:
            stmt = stmt.where(Chunk.node_id == node_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, chunk_id: UUID, data: ChunkUpdate) -> Chunk | None:
        chunk = await self.get(chunk_id)
        if not chunk:
            return None
        if data.text is not None:
            chunk.text = data.text
            vectors = await self.embeddings.embed_texts([data.text])
            chunk.embedding = vectors[0]
        if data.props is not None:
            chunk.props = data.props
        await self.session.flush()
        await self.session.refresh(chunk)
        return chunk

    async def delete(self, chunk_id: UUID) -> bool:
        chunk = await self.get(chunk_id)
        if not chunk:
            return False
        await self.session.delete(chunk)
        await self.session.flush()
        return True
