from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Select, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from graphrag.adapters.db.models import Chunk, Edge, Node
from graphrag.api.schemas import EdgeOut, NodeOut, SearchHit, SearchResponse, SubgraphOut
from graphrag.config import Settings
from graphrag.domain.graph import next_frontier
from graphrag.domain.models import GraphEdge
from graphrag.services.embedding_service import EmbeddingService


@dataclass
class RankedHit:
    chunk: Chunk
    node: Node
    score: float
    hop: int


class RetrievalService:
    def __init__(
        self,
        session: AsyncSession,
        embeddings: EmbeddingService,
        settings: Settings,
    ) -> None:
        self.session = session
        self.embeddings = embeddings
        self.settings = settings

    async def hybrid_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        node_types: list[str] | None = None,
        expand_hops: int | None = None,
        edge_types: list[str] | None = None,
    ) -> SearchResponse:
        top_k = top_k or self.settings.retrieval_top_k
        expand_hops = self.settings.expand_hops if expand_hops is None else expand_hops

        qvec = await self.embeddings.embed_query(query)
        await self.session.execute(
            text(f"SET LOCAL hnsw.ef_search = {int(self.settings.hnsw_ef_search)}")
        )

        seed_hits = await self._vector_search(qvec, top_k=top_k, node_types=node_types)
        ranked: dict[UUID, RankedHit] = {h.chunk.id: h for h in seed_hits}

        seed_node_ids = {h.node.id for h in seed_hits}
        subgraph_nodes: dict[UUID, Node] = {h.node.id: h.node for h in seed_hits}
        subgraph_edges: dict[UUID, Edge] = {}

        frontier = set(seed_node_ids)
        visited = set(seed_node_ids)

        for hop in range(1, expand_hops + 1):
            if not frontier:
                break
            edges = await self._fetch_edges(frontier, edge_types=edge_types)
            for edge in edges:
                subgraph_edges[edge.id] = edge

            domain_edges = [
                GraphEdge(
                    id=e.id,
                    src_id=e.src_id,
                    dst_id=e.dst_id,
                    type=e.type,
                    props=e.props or {},
                )
                for e in edges
            ]
            frontier = next_frontier(domain_edges, frontier, visited)
            visited |= frontier

            if not frontier:
                break

            nodes = await self._fetch_nodes(frontier, node_types=node_types)
            for node in nodes:
                subgraph_nodes[node.id] = node

            neighbor_chunks = await self._chunks_for_nodes([n.id for n in nodes])
            base = max((h.score for h in seed_hits), default=0.5)
            decayed = base * (0.5**hop)
            for chunk in neighbor_chunks:
                node = subgraph_nodes.get(chunk.node_id)
                if not node:
                    continue
                existing = ranked.get(chunk.id)
                if existing is None or decayed > existing.score:
                    ranked[chunk.id] = RankedHit(chunk=chunk, node=node, score=decayed, hop=hop)

        ordered = sorted(ranked.values(), key=lambda h: h.score, reverse=True)
        hits = [
            SearchHit(
                chunk_id=h.chunk.id,
                node_id=h.node.id,
                node_name=h.node.name,
                node_type=h.node.type,
                text=h.chunk.text,
                score=float(h.score),
                hop=h.hop,
            )
            for h in ordered
        ]

        return SearchResponse(
            hits=hits,
            subgraph=SubgraphOut(
                nodes=[NodeOut.model_validate(n) for n in subgraph_nodes.values()],
                edges=[EdgeOut.model_validate(e) for e in subgraph_edges.values()],
            ),
        )

    async def _vector_search(
        self,
        qvec: list[float],
        *,
        top_k: int,
        node_types: list[str] | None,
    ) -> list[RankedHit]:
        distance = Chunk.embedding.cosine_distance(qvec)
        stmt: Select = (
            select(Chunk, Node, (1 - distance).label("score"))
            .join(Node, Node.id == Chunk.node_id)
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )
        if node_types:
            stmt = stmt.where(Node.type.in_(node_types))
        rows = (await self.session.execute(stmt)).all()
        return [
            RankedHit(chunk=chunk, node=node, score=float(score), hop=0)
            for chunk, node, score in rows
        ]

    async def _fetch_edges(
        self,
        node_ids: set[UUID],
        *,
        edge_types: list[str] | None,
    ) -> list[Edge]:
        stmt = select(Edge).where(
            (Edge.src_id.in_(node_ids)) | (Edge.dst_id.in_(node_ids))
        )
        if edge_types:
            stmt = stmt.where(Edge.type.in_(edge_types))
        return list((await self.session.execute(stmt)).scalars().all())

    async def _fetch_nodes(
        self,
        node_ids: set[UUID],
        *,
        node_types: list[str] | None,
    ) -> list[Node]:
        stmt = select(Node).where(Node.id.in_(node_ids))
        if node_types:
            stmt = stmt.where(Node.type.in_(node_types))
        return list((await self.session.execute(stmt)).scalars().all())

    async def _chunks_for_nodes(self, node_ids: list[UUID]) -> list[Chunk]:
        if not node_ids:
            return []
        stmt = select(Chunk).where(Chunk.node_id.in_(node_ids)).options(selectinload(Chunk.node))
        return list((await self.session.execute(stmt)).scalars().all())


def pack_context(hits: list[SearchHit], token_budget: int) -> str:
    """Pack ranked hits into a labeled context string within a rough token budget."""
    char_budget = max(token_budget, 1) * 4
    parts: list[str] = []
    used = 0
    seen_nodes: set[UUID] = set()

    for hit in hits:
        block = f"[node:{hit.node_name}|{hit.node_type}|chunk:{hit.chunk_id}]\n{hit.text}"
        cost = len(block)
        if used + cost > char_budget and parts:
            break
        parts.append(block)
        used += cost
        seen_nodes.add(hit.node_id)
    return "\n\n".join(parts)
