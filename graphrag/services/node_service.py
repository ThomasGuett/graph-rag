from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Edge, Node
from graphrag.api.schemas import NodeCreate, NodeUpdate, SubgraphOut
from graphrag.api.schemas import EdgeOut, NodeOut


class NodeService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, data: NodeCreate) -> Node:
        node = Node(type=data.type.strip(), name=data.name.strip(), props=data.props or {})
        self.session.add(node)
        await self.session.flush()
        await self.session.refresh(node)
        return node

    async def get(self, node_id: UUID) -> Node | None:
        return await self.session.get(Node, node_id)

    async def list(
        self,
        *,
        type: str | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Node]:
        stmt = select(Node).order_by(Node.created_at.desc()).limit(limit).offset(offset)
        if type:
            stmt = stmt.where(Node.type == type)
        if q:
            stmt = stmt.where(Node.name.ilike(f"%{q}%"))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(self, node_id: UUID, data: NodeUpdate) -> Node | None:
        node = await self.get(node_id)
        if not node:
            return None
        if data.type is not None:
            node.type = data.type.strip()
        if data.name is not None:
            node.name = data.name.strip()
        if data.props is not None:
            node.props = data.props
        await self.session.flush()
        await self.session.refresh(node)
        return node

    async def delete(self, node_id: UUID) -> bool:
        node = await self.get(node_id)
        if not node:
            return False
        await self.session.delete(node)
        await self.session.flush()
        return True

    async def neighbors(
        self,
        node_id: UUID,
        *,
        direction: str = "both",
        edge_type: str | None = None,
        depth: int = 1,
    ) -> SubgraphOut:
        node = await self.get(node_id)
        if not node:
            return SubgraphOut()

        visited_nodes: dict[UUID, Node] = {node.id: node}
        collected_edges: dict[UUID, Edge] = {}
        frontier = {node.id}

        for _ in range(max(depth, 0)):
            if not frontier:
                break
            edge_stmt = select(Edge)
            if direction == "out":
                edge_stmt = edge_stmt.where(Edge.src_id.in_(frontier))
            elif direction == "in":
                edge_stmt = edge_stmt.where(Edge.dst_id.in_(frontier))
            else:
                edge_stmt = edge_stmt.where(
                    (Edge.src_id.in_(frontier)) | (Edge.dst_id.in_(frontier))
                )
            if edge_type:
                edge_stmt = edge_stmt.where(Edge.type == edge_type)

            edges = list((await self.session.execute(edge_stmt)).scalars().all())
            next_frontier: set[UUID] = set()
            for edge in edges:
                collected_edges[edge.id] = edge
                for endpoint in (edge.src_id, edge.dst_id):
                    if endpoint not in visited_nodes:
                        next_frontier.add(endpoint)
            frontier = next_frontier - set(visited_nodes.keys())
            if frontier:
                nodes = list(
                    (await self.session.execute(select(Node).where(Node.id.in_(frontier)))).scalars().all()
                )
                for n in nodes:
                    visited_nodes[n.id] = n

        return SubgraphOut(
            nodes=[NodeOut.model_validate(n) for n in visited_nodes.values()],
            edges=[EdgeOut.model_validate(e) for e in collected_edges.values()],
        )
