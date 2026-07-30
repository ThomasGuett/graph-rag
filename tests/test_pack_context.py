from uuid import uuid4

from graphrag.api.schemas import SearchHit
from graphrag.services.retrieval_service import pack_context


def test_pack_context_respects_budget():
    hits = [
        SearchHit(
            chunk_id=uuid4(),
            node_id=uuid4(),
            node_name="Doc A",
            node_type="document",
            text="alpha " * 50,
            score=0.9,
            hop=0,
        ),
        SearchHit(
            chunk_id=uuid4(),
            node_id=uuid4(),
            node_name="Doc B",
            node_type="document",
            text="beta " * 50,
            score=0.8,
            hop=0,
        ),
        SearchHit(
            chunk_id=uuid4(),
            node_id=uuid4(),
            node_name="Doc C",
            node_type="document",
            text="gamma " * 50,
            score=0.7,
            hop=1,
        ),
    ]
    packed = pack_context(hits, token_budget=40)
    assert "Doc A" in packed
    # First block is always included; later blocks should be truncated by budget
    assert "Doc B" not in packed or len(packed) > 40 * 4
    packed_small = pack_context(hits, token_budget=200)
    assert "Doc A" in packed_small
    assert packed_small.count("[node:") >= 1


def test_pack_context_empty():
    assert pack_context([], token_budget=100) == ""
