"""Entity seed lookup for local search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from graphrag.adapters.db.models import Chunk, Node
from graphrag.services.entity_resolution_service import normalize_entity_name

_SCAFFOLD_TYPES = frozenset({"document", "community"})
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,}", re.IGNORECASE)


@dataclass(slots=True)
class EntitySeed:
    node: Node
    score: float
    match: str  # exact | alias | embedding


def query_candidate_phrases(query: str) -> list[str]:
    """Build normalized phrases to match against entity names/aliases."""
    normalized = normalize_entity_name(query)
    phrases = {normalized}
    tokens = _TOKEN_RE.findall(normalized)
    phrases.update(tokens)
    # Adjacent bigrams help with multi-word entity names.
    for i in range(len(tokens) - 1):
        phrases.add(f"{tokens[i]} {tokens[i + 1]}")
    return [p for p in phrases if p]


async def lookup_entity_seeds(
    session: AsyncSession,
    query: str,
    *,
    qvec: list[float] | None,
    top_k: int,
    score_threshold: float,
) -> list[EntitySeed]:
    """Resolve entity seeds via exact/alias match, then entity-description ANN."""
    by_id: dict[UUID, EntitySeed] = {}

    for seed in await _exact_and_alias_seeds(session, query):
        existing = by_id.get(seed.node.id)
        if existing is None or seed.score > existing.score:
            by_id[seed.node.id] = seed

    if qvec is not None:
        for seed in await _embedding_seeds(session, qvec, top_k=top_k * 2):
            if seed.score < score_threshold:
                continue
            existing = by_id.get(seed.node.id)
            if existing is None or seed.score > existing.score:
                by_id[seed.node.id] = seed

    ranked = sorted(by_id.values(), key=lambda s: s.score, reverse=True)
    return ranked[:top_k]


async def _exact_and_alias_seeds(session: AsyncSession, query: str) -> list[EntitySeed]:
    phrases = query_candidate_phrases(query)
    if not phrases:
        return []

    stmt = select(Node).where(Node.type.notin_(list(_SCAFFOLD_TYPES)))
    nodes = list((await session.execute(stmt)).scalars().all())
    seeds: list[EntitySeed] = []
    phrase_set = set(phrases)
    for node in nodes:
        name_norm = normalize_entity_name(node.name)
        props = node.props or {}
        stored_norm = normalize_entity_name(str(props.get("normalized_name") or name_norm))
        aliases = [
            normalize_entity_name(str(a))
            for a in (props.get("aliases") or [])
            if a
        ]
        if stored_norm in phrase_set or name_norm in phrase_set:
            seeds.append(EntitySeed(node=node, score=1.0, match="exact"))
            continue
        if any(alias in phrase_set for alias in aliases):
            seeds.append(EntitySeed(node=node, score=0.95, match="alias"))
            continue
        # Substring containment for longer names inside the query.
        if len(name_norm) >= 3 and name_norm in normalize_entity_name(query):
            seeds.append(EntitySeed(node=node, score=0.9, match="exact"))
    return seeds


async def _embedding_seeds(
    session: AsyncSession,
    qvec: list[float],
    *,
    top_k: int,
) -> list[EntitySeed]:
    distance = Chunk.embedding.cosine_distance(qvec)
    stmt: Select = (
        select(Chunk, Node, (1 - distance).label("score"))
        .join(Node, Node.id == Chunk.node_id)
        .where(Chunk.embedding.is_not(None))
        .where(Chunk.props.contains({"kind": "entity_description"}))
        .where(Node.type.notin_(list(_SCAFFOLD_TYPES)))
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [
        EntitySeed(node=node, score=float(score), match="embedding")
        for _chunk, node, score in rows
    ]


def looks_thematic(query: str) -> bool:
    q = query.lower()
    keywords = (
        "overview",
        "theme",
        "themes",
        "across",
        "overall",
        "summary",
        "summarize",
        "high-level",
        "high level",
        "broadly",
        "in general",
        "what are the main",
        "main topics",
        "corpus",
        "entire",
        "whole dataset",
    )
    return any(k in q for k in keywords)


async def communities_exist(session: AsyncSession) -> bool:
    stmt = select(func.count()).select_from(Node).where(Node.type == "community")
    return int((await session.execute(stmt)).scalar_one()) > 0
