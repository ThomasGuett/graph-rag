"""Graph traversal helpers independent of persistence."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from uuid import UUID

from graphrag.domain.models import GraphEdge


def other_endpoint(edge: GraphEdge, node_id: UUID) -> UUID | None:
    if edge.src_id == node_id:
        return edge.dst_id
    if edge.dst_id == node_id:
        return edge.src_id
    return None


def filter_edges_for_direction(
    edges: Iterable[GraphEdge],
    frontier: set[UUID],
    *,
    direction: str = "both",
) -> list[GraphEdge]:
    """Keep edges incident to frontier according to direction (in|out|both)."""
    kept: list[GraphEdge] = []
    for edge in edges:
        if direction == "out":
            if edge.src_id in frontier:
                kept.append(edge)
        elif direction == "in":
            if edge.dst_id in frontier:
                kept.append(edge)
        elif edge.src_id in frontier or edge.dst_id in frontier:
            kept.append(edge)
    return kept


def next_frontier(
    edges: Iterable[GraphEdge],
    frontier: set[UUID],
    visited: set[UUID],
) -> set[UUID]:
    """Return neighbor node ids reachable in one hop, excluding visited."""
    nxt: set[UUID] = set()
    for edge in edges:
        for node_id in frontier:
            other = other_endpoint(edge, node_id)
            if other is not None and other not in visited:
                nxt.add(other)
    return nxt


def bfs_node_ids(
    seed_ids: Iterable[UUID],
    adjacency: dict[UUID, list[UUID]],
    *,
    max_hops: int,
) -> dict[UUID, int]:
    """
    BFS over an adjacency map.

    Returns mapping of node_id -> hop distance (0 for seeds).
    """
    hops: dict[UUID, int] = {}
    queue: deque[UUID] = deque()
    for seed in seed_ids:
        hops[seed] = 0
        queue.append(seed)

    while queue:
        current = queue.popleft()
        current_hop = hops[current]
        if current_hop >= max_hops:
            continue
        for neighbor in adjacency.get(current, []):
            if neighbor in hops:
                continue
            hops[neighbor] = current_hop + 1
            queue.append(neighbor)
    return hops


def build_undirected_adjacency(edges: Iterable[GraphEdge]) -> dict[UUID, list[UUID]]:
    adj: dict[UUID, list[UUID]] = {}
    for edge in edges:
        adj.setdefault(edge.src_id, []).append(edge.dst_id)
        adj.setdefault(edge.dst_id, []).append(edge.src_id)
    return adj
