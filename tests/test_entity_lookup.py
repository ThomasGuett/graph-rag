from graphrag.services.entity_lookup import looks_thematic, query_candidate_phrases


def test_query_candidate_phrases_includes_tokens_and_bigrams():
    phrases = query_candidate_phrases("Dr. Smith at Boston General")
    joined = " ".join(phrases)
    assert "boston" in phrases
    assert "general" in phrases
    assert "boston general" in phrases
    assert "smith" in joined


def test_looks_thematic_true():
    assert looks_thematic("Give an overview of the main themes across the corpus")
    assert looks_thematic("What are the overall topics?")


def test_looks_thematic_false():
    assert not looks_thematic("Who leads oncology at Boston General?")
