"""Flat connected-component communities + LLM summaries."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Chunk, Community, CommunityMember, Edge, Node
from graphrag.adapters.llm.base import LLMClient
from graphrag.config import Settings
from graphrag.services.embedding_service import EmbeddingService

_SCAFFOLD_TYPES = frozenset({"document", "community"})

_SUMMARY_SYSTEM = """You summarize a cluster of related knowledge-graph entities.
Write 2-4 sentences covering the main theme and key relationships.
Return plain text only, no markdown."""


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[UUID, UUID] = {}
        self.rank: dict[UUID, int] = {}

    def add(self, x: UUID) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: UUID) -> UUID:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: UUID, b: UUID) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def components(self) -> dict[UUID, list[UUID]]:
        groups: dict[UUID, list[UUID]] = defaultdict(list)
        for node_id in self.parent:
            groups[self.find(node_id)].append(node_id)
        return dict(groups)


def connected_components(
    node_ids: list[UUID],
    edges: list[tuple[UUID, UUID]],
) -> list[list[UUID]]:
    uf = UnionFind()
    for nid in node_ids:
        uf.add(nid)
    for src, dst in edges:
        if src in uf.parent and dst in uf.parent:
            uf.union(src, dst)
    return [members for members in uf.components().values() if members]


class CommunityService:
    def __init__(
        self,
        session: AsyncSession,
        llm: LLMClient,
        embeddings: EmbeddingService,
        settings: Settings,
    ) -> None:
        self.session = session
        self.llm = llm
        self.embeddings = embeddings
        self.settings = settings

    async def rebuild(self) -> list[Community]:
        await self._clear_existing()

        nodes = list(
            (
                await self.session.execute(
                    select(Node).where(Node.type.notin_(list(_SCAFFOLD_TYPES)))
                )
            )
            .scalars()
            .all()
        )
        node_ids = [n.id for n in nodes]
        node_by_id = {n.id: n for n in nodes}
        if not node_ids:
            return []

        edge_rows = list(
            (
                await self.session.execute(
                    select(Edge).where(
                        Edge.src_id.in_(node_ids),
                        Edge.dst_id.in_(node_ids),
                        Edge.type != "mentions",
                    )
                )
            )
            .scalars()
            .all()
        )
        undirected = [(e.src_id, e.dst_id) for e in edge_rows]
        components = connected_components(node_ids, undirected)
        min_size = self.settings.community_min_size

        communities: list[Community] = []
        for i, member_ids in enumerate(sorted(components, key=len, reverse=True)):
            if len(member_ids) < min_size:
                continue
            members = [node_by_id[mid] for mid in member_ids if mid in node_by_id]
            if not members:
                continue
            label = f"community_{i + 1}"
            summary = await self._summarize(members, edge_rows)
            community_node = Node(
                type="community",
                name=label,
                props={"member_count": len(members)},
            )
            self.session.add(community_node)
            await self.session.flush()

            vectors = await self.embeddings.embed_texts([summary])
            self.session.add(
                Chunk(
                    node_id=community_node.id,
                    text=summary,
                    embedding=vectors[0],
                    props={"kind": "community_summary"},
                )
            )

            community = Community(
                label=label,
                summary=summary,
                node_id=community_node.id,
                member_count=len(members),
                props={},
            )
            self.session.add(community)
            await self.session.flush()
            for member in members:
                self.session.add(
                    CommunityMember(community_id=community.id, node_id=member.id)
                )
            communities.append(community)

        await self.session.flush()
        for c in communities:
            await self.session.refresh(c)
        return communities

    async def list(self, *, limit: int = 100, offset: int = 0) -> list[Community]:
        stmt = (
            select(Community)
            .order_by(Community.member_count.desc(), Community.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get(self, community_id: UUID) -> Community | None:
        return await self.session.get(Community, community_id)

    async def members(self, community_id: UUID) -> list[Node]:
        stmt = (
            select(Node)
            .join(CommunityMember, CommunityMember.node_id == Node.id)
            .where(CommunityMember.community_id == community_id)
            .order_by(Node.name)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def _clear_existing(self) -> None:
        old = list((await self.session.execute(select(Community))).scalars().all())
        community_node_ids = [c.node_id for c in old if c.node_id]
        await self.session.execute(delete(CommunityMember))
        await self.session.execute(delete(Community))
        if community_node_ids:
            await self.session.execute(delete(Node).where(Node.id.in_(community_node_ids)))
        await self.session.flush()

    async def _summarize(self, members: list[Node], edges: list[Edge]) -> str:
        member_ids = {m.id for m in members}
        lines = [f"- {m.name} ({m.type})" for m in members[:40]]
        for m in members[:20]:
            desc = str((m.props or {}).get("description") or "").strip()
            if desc:
                lines.append(f"  desc[{m.name}]: {desc[:240]}")
        rel_lines = []
        name_by_id = {m.id: m.name for m in members}
        for e in edges:
            if e.src_id in member_ids and e.dst_id in member_ids:
                rel_lines.append(
                    f"- {name_by_id.get(e.src_id)} -[{e.type}]-> {name_by_id.get(e.dst_id)}"
                )
                if len(rel_lines) >= 40:
                    break
        user = (
            "Entities:\n"
            + "\n".join(lines)
            + "\n\nRelationships:\n"
            + ("\n".join(rel_lines) if rel_lines else "(none)")
        )
        return (await self.llm.complete(system=_SUMMARY_SYSTEM, user=user, temperature=0.2)).strip()
