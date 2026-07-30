from graphrag.services.chunking_service import chunk_text


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_single_paragraph():
    spans = chunk_text("Hello world.", chunk_size=100, chunk_overlap=10)
    assert len(spans) == 1
    assert spans[0].text == "Hello world."
    assert spans[0].ord == 0
    assert spans[0].props["kind"] == "document_chunk"


def test_chunk_text_packs_paragraphs():
    text = "Para one.\n\nPara two.\n\nPara three."
    spans = chunk_text(text, chunk_size=40, chunk_overlap=5)
    assert len(spans) >= 1
    joined = " ".join(s.text for s in spans)
    assert "Para one" in joined
    assert "Para three" in joined
    assert all(s.char_start <= s.char_end for s in spans)


def test_chunk_text_hard_splits_long_paragraph():
    text = "x" * 2500
    spans = chunk_text(text, chunk_size=1000, chunk_overlap=100)
    assert len(spans) >= 3
    assert all(len(s.text) <= 1000 + 100 + 2 for s in spans)  # allow overlap prefix + newlines


def test_chunk_overlap_must_be_less_than_size():
    try:
        chunk_text("abc", chunk_size=10, chunk_overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
