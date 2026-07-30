from graphrag.services.extraction_service import parse_extraction_json


def test_parse_extraction_json_happy_path():
    raw = """
    {
      "entities": [
        {"name": "Dr. Smith", "type": "Person", "description": "Oncologist"},
        {"name": "Boston General", "type": "org", "description": "Hospital"}
      ],
      "relationships": [
        {"source": "Dr. Smith", "target": "Boston General", "type": "works_at", "description": ""}
      ]
    }
    """
    result = parse_extraction_json(raw)
    assert len(result.entities) == 2
    assert result.entities[0].type == "person"
    assert result.relationships[0].type == "works_at"


def test_parse_extraction_json_fenced():
    raw = """```json
{"entities": [{"name": "A", "type": "concept"}], "relationships": []}
```"""
    result = parse_extraction_json(raw)
    assert len(result.entities) == 1
    assert result.entities[0].name == "A"


def test_parse_extraction_skips_self_loops():
    raw = '{"entities": [{"name": "A", "type": "x"}], "relationships": [{"source": "A", "target": "A", "type": "same"}]}'
    result = parse_extraction_json(raw)
    assert result.relationships == []
