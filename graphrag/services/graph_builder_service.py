"""Write extracted entities/relationships into the graph."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Chunk, Edge, Node
from graphrag.services.embedding_service import EmbeddingService
from graphrag.services.entity_resolution_service import (
    EntityResolutionService,
    normalize_entity_name,
)
from graphrag.services.extraction_service import ExtractionResult


class GraphBuilderService:
    def __init__(
        self,
        session: AsyncSession,
        resolver: EntityResolutionService,
        embeddings: EmbeddingService,
    ) -> None:
        self.session = session
        self.resolver = resolver
        self.embeddings = embeddings

    async def apply_extraction(
        self,
        *,
        document_node_id: UUID,
        chunk_id: UUID | None,
        extraction: ExtractionResult,
    ) -> dict[str, int]:
        resolved = await self.resolver.resolve_many(extraction.entities)
        mentions = 0
        for node in resolved.values():
            created = await self._ensure_edge(
                src_id=document_node_id,
                dst_id=node.id,
                etype="mentions",
                props={"chunk_id": str(chunk_id)} if chunk_id else {},
            )
            if created:
                mentions += 1

        rels = 0
        for rel in extraction.relationships:
            src = self._lookup(resolved, rel.source)
            dst = self._lookup(resolved, rel.target)
            if not src or not dst or src.id == dst.id:
                continue
            created = await self._ensure_edge(
                src_id=src.id,
                dst_id=dst.id,
                etype=rel.type,
                props={"description": rel.description} if rel.description else {},
            )
            if created:
                rels += 1

        desc_chunks = await self._ensure_description_chunks(list(resolved.values()))
        return {
            "entities": len(resolved),
            "mentions": mentions,
            "relationships": rels,
            "description_chunks": desc_chunks,
        }

    def _lookup(self, resolved: dict[tuple[str, str], Node], name: str) -> Node | None:
        needle = normalize_entity_name(name)
        for (_etype, nname), node in resolved.items():
            if nname == needle:
                return node
        # Fall back to resolver cache across types.
        for (_etype, nname), node in self.resolver._cache.items():
            if nname == needle:
                return node
        return None

    async def _ensure_edge(
        self,
        *,
        src_id: UUID,
        dst_id: UUID,
        etype: str,
        props: dict,
    ) -> bool:
        existing = await self.session.execute(
            select(Edge).where(
                Edge.src_id == src_id,
                Edge.dst_id == dst_id,
                Edge.type == etype,
            )
        )
        edge = existing.scalar_one_or_none()
        if edge:
            if props:
                merged = dict(edge.props or {})
                merged.update({k: v for k, v in props.items() if v})
                edge.props = merged
                await self.session.flush()
            return False
        edge = Edge(src_id=src_id, dst_id=dst_id, type=etype, props=props or {})
        self.session.add(edge)
        await self.session.flush()
        return True

    async def _ensure_description_chunks(self, nodes: list[Node]) -> int:
        created = 0
        to_embed: list[tuple[Node, str]] = []
        for node in nodes:
            desc = str((node.props or {}).get("description") or "").strip()
            if not desc:
                continue
            existing = await self.session.execute(
                select(Chunk).where(
                    Chunk.node_id == node.id,
                    Chunk.props.contains({"kind": "entity_description"}),
                )
            )
            chunk = existing.scalar_one_or_none()
            if chunk:
                if chunk.text != desc:
                    chunk.text = desc
                    to_embed.append((node, desc))
                    chunk.props = {**(chunk.props or {}), "kind": "entity_description"}
                continue
            to_embed.append((node, desc))

        if not to_embed:
            return 0

        texts = [t for _, t in to_embed]
        vectors = await self.embeddings.embed_texts(texts)
        for (node, desc), vector in zip(to_embed, vectors, strict=True):
            existing = await self.session.execute(
                select(Chunk).where(
                    Chunk.node_id == node.id,
                    Chunk.props.contains({"kind": "entity_description"}),
                )
            )
            chunk = existing.scalar_one_or_none()
            if chunk:
                chunk.text = desc
                chunk.embedding = vector
            else:
                self.session.add(
                    Chunk(
                        node_id=node.id,
                        text=desc,
                        embedding=vector,
                        props={"kind": "entity_description"},
                    )
                )
                created += 1
        await self.session.flush()
        return created
