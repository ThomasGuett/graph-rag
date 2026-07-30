from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Select, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from graphrag.adapters.db.models import Chunk, Edge, Node
from graphrag.api.schemas import (
    EdgeOut,
    NodeOut,
    SearchHit,
    SearchMode,
    SearchResponse,
    SubgraphOut,
)
from graphrag.config import Settings
from graphrag.domain.graph import next_frontier
from graphrag.domain.models import GraphEdge
from graphrag.services.embedding_service import EmbeddingService
from graphrag.services.entity_lookup import (
    communities_exist,
    lookup_entity_seeds,
    looks_thematic,
)


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

    async def search(
        self,
        query: str,
        *,
        mode: SearchMode = "auto",
        top_k: int | None = None,
        node_types: list[str] | None = None,
        expand_hops: int | None = None,
        edge_types: list[str] | None = None,
    ) -> SearchResponse:
        resolved = await self.resolve_mode(query, mode)
        if resolved == "local":
            return await self.local_search(
                query,
                top_k=top_k,
                expand_hops=expand_hops,
                edge_types=edge_types,
            )
        if resolved == "global":
            return await self.global_search(query, top_k=top_k)
        return await self.hybrid_search(
            query,
            top_k=top_k,
            node_types=node_types,
            expand_hops=expand_hops,
            edge_types=edge_types,
        )

    async def resolve_mode(self, query: str, mode: SearchMode) -> SearchMode:
        if mode != "auto":
            return mode

        qvec = await self.embeddings.embed_query(query)
        seeds = await lookup_entity_seeds(
            self.session,
            query,
            qvec=qvec,
            top_k=self.settings.local_entity_top_k,
            score_threshold=self.settings.auto_entity_score_threshold,
        )
        strong = [s for s in seeds if s.score >= self.settings.auto_entity_score_threshold]
        thematic = looks_thematic(query)
        has_communities = await communities_exist(self.session)

        if strong and not thematic:
            return "local"
        if has_communities and (thematic or not strong):
            return "global"
        return "hybrid"

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
            edges = await self._fetch_edges(frontier, edge_types=edge_types, exclude_types=None)
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

        return self._to_response(ranked, subgraph_nodes, subgraph_edges, mode_used="hybrid")

    async def local_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        expand_hops: int | None = None,
        edge_types: list[str] | None = None,
    ) -> SearchResponse:
        top_k = top_k or self.settings.retrieval_top_k
        expand_hops = self.settings.expand_hops if expand_hops is None else expand_hops

        qvec = await self.embeddings.embed_query(query)
        await self.session.execute(
            text(f"SET LOCAL hnsw.ef_search = {int(self.settings.hnsw_ef_search)}")
        )

        seeds = await lookup_entity_seeds(
            self.session,
            query,
            qvec=qvec,
            top_k=self.settings.local_entity_top_k,
            score_threshold=self.settings.auto_entity_score_threshold,
        )
        if not seeds:
            response = await self.hybrid_search(
                query, top_k=top_k, expand_hops=expand_hops, edge_types=edge_types
            )
            response.mode_used = "hybrid"
            return response

        seed_scores = {s.node.id: s.score for s in seeds}
        seed_node_ids = set(seed_scores)
        subgraph_nodes: dict[UUID, Node] = {s.node.id: s.node for s in seeds}
        subgraph_edges: dict[UUID, Edge] = {}
        hop_by_node: dict[UUID, int] = {nid: 0 for nid in seed_node_ids}

        frontier = set(seed_node_ids)
        visited = set(seed_node_ids)

        for hop in range(1, expand_hops + 1):
            if not frontier:
                break
            # Expand on typed entity edges only (exclude mentions scaffolding).
            if edge_types:
                expand_types = [t for t in edge_types if t != "mentions"]
            else:
                expand_types = None
            edges = await self._fetch_edges(
                frontier,
                edge_types=expand_types,
                exclude_types=["mentions"] if expand_types is None else None,
            )
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
            for nid in frontier:
                hop_by_node[nid] = hop

            if not frontier:
                break
            nodes = await self._fetch_nodes(frontier, node_types=None)
            for node in nodes:
                if node.type in {"document", "community"}:
                    continue
                subgraph_nodes[node.id] = node

        ranked: dict[UUID, RankedHit] = {}

        # Entity description chunks on visited entity nodes.
        entity_ids = [
            nid
            for nid, node in subgraph_nodes.items()
            if node.type not in {"document", "community"}
        ]
        entity_chunks = await self._chunks_for_nodes(entity_ids)
        for chunk in entity_chunks:
            node = subgraph_nodes.get(chunk.node_id)
            if not node:
                continue
            hop = hop_by_node.get(node.id, 0)
            seed_score = seed_scores.get(node.id)
            if seed_score is None:
                # Inherit max seed score among seeds that reached this node via hop decay base.
                seed_score = max(seed_scores.values(), default=0.5)
            score = float(seed_score) * (0.5**hop)
            # Prefer entity_description chunks slightly.
            if (chunk.props or {}).get("kind") == "entity_description":
                score *= 1.05
            existing = ranked.get(chunk.id)
            if existing is None or score > existing.score:
                ranked[chunk.id] = RankedHit(chunk=chunk, node=node, score=score, hop=hop)

        # Document evidence via reverse mentions / chunk_id props.
        mention_edges = await self._fetch_mentions_into(entity_ids)
        doc_node_ids: set[UUID] = set()
        chunk_ids_from_props: set[UUID] = set()
        for edge in mention_edges:
            subgraph_edges[edge.id] = edge
            doc_node_ids.add(edge.src_id)
            raw_chunk = (edge.props or {}).get("chunk_id")
            if raw_chunk:
                try:
                    chunk_ids_from_props.add(UUID(str(raw_chunk)))
                except ValueError:
                    pass

        if doc_node_ids:
            doc_nodes = await self._fetch_nodes(doc_node_ids, node_types=None)
            for node in doc_nodes:
                subgraph_nodes[node.id] = node

        # Prefer chunks referenced on mentions; also include document node chunks.
        evidence_chunks: list[Chunk] = []
        if chunk_ids_from_props:
            evidence_chunks.extend(await self._chunks_by_ids(chunk_ids_from_props))
        evidence_chunks.extend(await self._chunks_for_nodes(list(doc_node_ids)))

        seen: set[UUID] = set()
        unique_evidence: list[Chunk] = []
        for chunk in evidence_chunks:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            unique_evidence.append(chunk)

        for chunk in unique_evidence:
            node = subgraph_nodes.get(chunk.node_id)
            if not node:
                continue
            # Document evidence hops from nearest mentioned entity (approx 1).
            hop = 1
            seed_score = max(seed_scores.values(), default=0.5)
            score = float(seed_score) * (0.5**hop)
            existing = ranked.get(chunk.id)
            if existing is None or score > existing.score:
                ranked[chunk.id] = RankedHit(chunk=chunk, node=node, score=score, hop=hop)

        ordered = sorted(ranked.values(), key=lambda h: h.score, reverse=True)[:top_k]
        ranked = {h.chunk.id: h for h in ordered}
        return self._to_response(ranked, subgraph_nodes, subgraph_edges, mode_used="local")

    async def global_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> SearchResponse:
        limit = top_k or self.settings.global_map_top_k
        qvec = await self.embeddings.embed_query(query)
        await self.session.execute(
            text(f"SET LOCAL hnsw.ef_search = {int(self.settings.hnsw_ef_search)}")
        )

        distance = Chunk.embedding.cosine_distance(qvec)
        stmt: Select = (
            select(Chunk, Node, (1 - distance).label("score"))
            .join(Node, Node.id == Chunk.node_id)
            .where(Chunk.embedding.is_not(None))
            .where(
                (Node.type == "community")
                | (Chunk.props.contains({"kind": "community_summary"}))
            )
            .order_by(distance)
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        ranked: dict[UUID, RankedHit] = {}
        subgraph_nodes: dict[UUID, Node] = {}
        for chunk, node, score in rows:
            ranked[chunk.id] = RankedHit(
                chunk=chunk, node=node, score=float(score), hop=0
            )
            subgraph_nodes[node.id] = node

        if not ranked:
            # Fall back to hybrid when no communities are indexed yet.
            response = await self.hybrid_search(query, top_k=limit)
            response.mode_used = "hybrid"
            return response

        return self._to_response(ranked, subgraph_nodes, {}, mode_used="global")

    def _to_response(
        self,
        ranked: dict[UUID, RankedHit],
        subgraph_nodes: dict[UUID, Node],
        subgraph_edges: dict[UUID, Edge],
        *,
        mode_used: SearchMode,
    ) -> SearchResponse:
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
            mode_used=mode_used,
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
        exclude_types: list[str] | None = None,
    ) -> list[Edge]:
        if not node_ids:
            return []
        stmt = select(Edge).where(
            (Edge.src_id.in_(node_ids)) | (Edge.dst_id.in_(node_ids))
        )
        if edge_types:
            stmt = stmt.where(Edge.type.in_(edge_types))
        if exclude_types:
            stmt = stmt.where(Edge.type.notin_(exclude_types))
        return list((await self.session.execute(stmt)).scalars().all())

    async def _fetch_mentions_into(self, entity_ids: list[UUID]) -> list[Edge]:
        if not entity_ids:
            return []
        stmt = select(Edge).where(
            Edge.type == "mentions",
            Edge.dst_id.in_(entity_ids),
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def _fetch_nodes(
        self,
        node_ids: set[UUID],
        *,
        node_types: list[str] | None,
    ) -> list[Node]:
        if not node_ids:
            return []
        stmt = select(Node).where(Node.id.in_(node_ids))
        if node_types:
            stmt = stmt.where(Node.type.in_(node_types))
        return list((await self.session.execute(stmt)).scalars().all())

    async def _chunks_for_nodes(self, node_ids: list[UUID]) -> list[Chunk]:
        if not node_ids:
            return []
        stmt = select(Chunk).where(Chunk.node_id.in_(node_ids)).options(selectinload(Chunk.node))
        return list((await self.session.execute(stmt)).scalars().all())

    async def _chunks_by_ids(self, chunk_ids: set[UUID]) -> list[Chunk]:
        if not chunk_ids:
            return []
        stmt = select(Chunk).where(Chunk.id.in_(chunk_ids))
        return list((await self.session.execute(stmt)).scalars().all())


def pack_context(hits: list[SearchHit], token_budget: int) -> str:
    """Pack ranked hits into a labeled context string within a rough token budget."""
    char_budget = max(token_budget, 1) * 4
    parts: list[str] = []
    used = 0

    for hit in hits:
        block = f"[node:{hit.node_name}|{hit.node_type}|chunk:{hit.chunk_id}]\n{hit.text}"
        cost = len(block)
        if used + cost > char_budget and parts:
            break
        parts.append(block)
        used += cost
    return "\n\n".join(parts)
