from arxiv_agent.chunking import chunk_text


class _WordTokenizer:
    """Whitespace tokenizer: fast, deterministic, no model download -
    exercises the packing/overlap/section algorithm in isolation from
    the real embedding model's tokenizer.
    """

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


def test_chunk_text_empty_returns_no_chunks():
    assert chunk_text("", _WordTokenizer(), target_tokens=10, overlap_tokens=2) == []


def test_chunk_text_whitespace_only_returns_no_chunks():
    assert chunk_text("   \n\n   \n\n  ", _WordTokenizer(), target_tokens=10, overlap_tokens=2) == []


def test_chunk_text_single_short_paragraph_is_one_chunk():
    chunks = chunk_text("one two three", _WordTokenizer(), target_tokens=10, overlap_tokens=2)
    assert len(chunks) == 1
    assert chunks[0].text == "one two three"
    assert chunks[0].section == ""


def test_chunk_text_splits_on_token_budget_and_tracks_sections():
    text = "# Intro\n\nAAA BBB CCC DDD\n\n## Method\n\nEEE FFF GGG HHH"
    chunks = chunk_text(text, _WordTokenizer(), target_tokens=6, overlap_tokens=0)
    assert len(chunks) == 2
    assert chunks[0].section == "Intro"
    assert "AAA BBB CCC DDD" in chunks[0].text
    assert chunks[1].section == "Method"
    assert "EEE FFF GGG HHH" in chunks[1].text


def test_chunk_text_overlap_repeats_trailing_paragraph():
    text = "AAA BBB CCC\n\nDDD EEE FFF\n\nGGG HHH III"
    chunks = chunk_text(text, _WordTokenizer(), target_tokens=6, overlap_tokens=3)
    assert len(chunks) == 2
    assert "DDD EEE FFF" in chunks[0].text
    assert "DDD EEE FFF" in chunks[1].text  # carried over as overlap
    assert "AAA BBB CCC" not in chunks[1].text
    assert "GGG HHH III" not in chunks[0].text


def test_chunk_text_hard_splits_a_single_oversized_paragraph():
    text = "AAA BBB CCC DDD EEE FFF GGG HHH"  # one paragraph, no blank lines, 8 tokens
    chunks = chunk_text(text, _WordTokenizer(), target_tokens=3, overlap_tokens=0)
    assert [c.text for c in chunks] == ["AAA BBB CCC", "DDD EEE FFF", "GGG HHH"]
    assert all(c.section == "" for c in chunks)


def test_chunk_text_strips_markdown_emphasis_from_section_title():
    text = "## **Abstract**\n\nsome text here"
    chunks = chunk_text(text, _WordTokenizer(), target_tokens=20, overlap_tokens=0)
    assert chunks[0].section == "Abstract"
