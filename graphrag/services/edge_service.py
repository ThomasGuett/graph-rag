from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Edge, Node
from graphrag.api.schemas import EdgeCreate, EdgeUpdate


class EdgeConflictError(Exception):
    pass


class EdgeService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: EdgeCreate) -> Edge:
        if data.src_id == data.dst_id:
            raise ValueError("src_id and dst_id must differ")
        src = await self.session.get(Node, data.src_id)
        dst = await self.session.get(Node, data.dst_id)
        if not src or not dst:
            raise LookupError("src_id or dst_id not found")

        edge = Edge(
            src_id=data.src_id,
            dst_id=data.dst_id,
            type=data.type.strip(),
            props=data.props or {},
        )
        self.session.add(edge)
        try:
            await self.session.flush()
        except IntegrityError as exc:
            raise EdgeConflictError("edge with same src/dst/type already exists") from exc
        await self.session.refresh(edge)
        return edge

    async def get(self, edge_id: UUID) -> Edge | None:
        return await self.session.get(Edge, edge_id)

    async def list(
        self,
        *,
        src_id: UUID | None = None,
        dst_id: UUID | None = None,
        type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Edge]:
        stmt = select(Edge).order_by(Edge.created_at.desc()).limit(limit).offset(offset)
        if src_id:
            stmt = stmt.where(Edge.src_id == src_id)
        if dst_id:
            stmt = stmt.where(Edge.dst_id == dst_id)
        if type:
            stmt = stmt.where(Edge.type == type)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, edge_id: UUID, data: EdgeUpdate) -> Edge | None:
        edge = await self.get(edge_id)
        if not edge:
            return None
        if data.type is not None:
            edge.type = data.type.strip()
        if data.props is not None:
            edge.props = data.props
        try:
            await self.session.flush()
        except IntegrityError as exc:
            raise EdgeConflictError("edge with same src/dst/type already exists") from exc
        await self.session.refresh(edge)
        return edge

    async def delete(self, edge_id: UUID) -> bool:
        edge = await self.get(edge_id)
        if not edge:
            return False
        await self.session.delete(edge)
        await self.session.flush()
        return True
