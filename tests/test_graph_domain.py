from uuid import uuid4

from graphrag.domain.graph import (
    bfs_node_ids,
    build_undirected_adjacency,
    filter_edges_for_direction,
    next_frontier,
    other_endpoint,
)
from graphrag.domain.models import GraphEdge


def _edge(src, dst, etype="related"):
    return GraphEdge(id=uuid4(), src_id=src, dst_id=dst, type=etype)


def test_other_endpoint():
    a, b = uuid4(), uuid4()
    e = _edge(a, b)
    assert other_endpoint(e, a) == b
    assert other_endpoint(e, b) == a
    assert other_endpoint(e, uuid4()) is None


def test_next_frontier_and_bfs():
    a, b, c = uuid4(), uuid4(), uuid4()
    edges = [_edge(a, b), _edge(b, c)]
    frontier = {a}
    visited = {a}
    nxt = next_frontier(edges, frontier, visited)
    assert nxt == {b}

    adj = build_undirected_adjacency(edges)
    hops = bfs_node_ids([a], adj, max_hops=2)
    assert hops[a] == 0
    assert hops[b] == 1
    assert hops[c] == 2


def test_filter_edges_direction():
    a, b, c = uuid4(), uuid4(), uuid4()
    edges = [_edge(a, b), _edge(c, a)]
    out_edges = filter_edges_for_direction(edges, {a}, direction="out")
    in_edges = filter_edges_for_direction(edges, {a}, direction="in")
    assert len(out_edges) == 1 and out_edges[0].dst_id == b
    assert len(in_edges) == 1 and in_edges[0].src_id == c
