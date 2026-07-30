from uuid import uuid4

from graphrag.services.community_service import connected_components
from graphrag.services.entity_resolution_service import normalize_entity_name, normalize_entity_type


def test_normalize_entity_name():
    assert normalize_entity_name("  Dr.   Smith ") == "dr. smith"


def test_normalize_entity_type():
    assert normalize_entity_type("Health-Org") == "health_org"


def test_connected_components_basic():
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    comps = connected_components([a, b, c, d], [(a, b), (b, c)])
    sizes = sorted(len(g) for g in comps)
    assert sizes == [1, 3]
    big = next(g for g in comps if len(g) == 3)
    assert set(big) == {a, b, c}
    small = next(g for g in comps if len(g) == 1)
    assert small == [d]


def test_connected_components_isolated():
    a, b = uuid4(), uuid4()
    comps = connected_components([a, b], [])
    assert sorted(len(g) for g in comps) == [1, 1]
