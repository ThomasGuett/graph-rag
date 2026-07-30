"""Upsert graph nodes by normalized (type, name)."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Node
from graphrag.services.extraction_service import ExtractedEntity


def normalize_entity_name(name: str) -> str:
    return " ".join(name.split()).strip().lower()


def normalize_entity_type(etype: str) -> str:
    return etype.strip().lower().replace(" ", "_").replace("-", "_") or "concept"


class EntityResolutionService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._cache: dict[tuple[str, str], Node] = {}

    async def resolve(self, entity: ExtractedEntity) -> Node:
        key = (normalize_entity_type(entity.type), normalize_entity_name(entity.name))
        if key in self._cache:
            node = self._cache[key]
            await self._merge_props(node, entity)
            return node

        existing = await self._find_existing(key[0], key[1])
        if existing:
            await self._merge_props(existing, entity)
            self._cache[key] = existing
            return existing

        props: dict = {}
        if entity.description:
            props["description"] = entity.description
        props["aliases"] = [entity.name]
        props["normalized_name"] = key[1]
        node = Node(type=key[0], name=entity.name.strip(), props=props)
        self.session.add(node)
        await self.session.flush()
        await self.session.refresh(node)
        self._cache[key] = node
        return node

    async def resolve_many(self, entities: list[ExtractedEntity]) -> dict[tuple[str, str], Node]:
        out: dict[tuple[str, str], Node] = {}
        for entity in entities:
            node = await self.resolve(entity)
            out[(normalize_entity_type(entity.type), normalize_entity_name(entity.name))] = node
        return out

    def get_cached(self, etype: str, name: str) -> Node | None:
        return self._cache.get((normalize_entity_type(etype), normalize_entity_name(name)))

    async def _find_existing(self, etype: str, normalized_name: str) -> Node | None:
        stmt = (
            select(Node)
            .where(Node.type == etype)
            .where(func.lower(func.trim(Node.name)) == normalized_name)
            .limit(1)
        )
        result = await self.session.execute(stmt)
        node = result.scalar_one_or_none()
        if node:
            return node
        # Also match previously stored normalized_name in props.
        stmt2 = (
            select(Node)
            .where(Node.type == etype)
            .where(Node.props.contains({"normalized_name": normalized_name}))
            .limit(1)
        )
        result2 = await self.session.execute(stmt2)
        return result2.scalar_one_or_none()

    async def _merge_props(self, node: Node, entity: ExtractedEntity) -> None:
        props = dict(node.props or {})
        aliases = list(props.get("aliases") or [])
        if entity.name not in aliases:
            aliases.append(entity.name)
        props["aliases"] = aliases
        props["normalized_name"] = normalize_entity_name(entity.name)
        if entity.description:
            existing_desc = str(props.get("description") or "").strip()
            if not existing_desc:
                props["description"] = entity.description
            elif entity.description not in existing_desc:
                props["description"] = f"{existing_desc}\n{entity.description}".strip()
        node.props = props
        await self.session.flush()
