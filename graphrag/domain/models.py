"""Pure domain types (no persistence / HTTP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(slots=True)
class GraphNode:
    id: UUID
    type: str
    name: str
    props: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class GraphEdge:
    id: UUID
    src_id: UUID
    dst_id: UUID
    type: str
    props: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(slots=True)
class TextChunk:
    id: UUID
    node_id: UUID
    text: str
    props: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class RankedChunk:
    chunk: TextChunk
    node: GraphNode
    score: float
    hop: int = 0
