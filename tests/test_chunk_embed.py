from arxiv_agent.nodes.chunk_embed import chunk_embed, route_after_chunk_embed
from arxiv_agent.schemas import ParsedPage, PaperMetadata
from arxiv_agent.state import create_initial_state


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


class _FakeModel:
    tokenizer = _FakeTokenizer()


class _FakeCollection:
    def __init__(self, name):
        self.name = name
        self.added = None

    def add(self, **kwargs):
        self.added = kwargs


def _make_paper() -> PaperMetadata:
    return PaperMetadata(
        title="Test Paper",
        authors=["A. Author"],
        arxiv_id="2401.12345",
        published="2024-01-15",
        link="https://arxiv.org/abs/2401.12345",
        abstract="abstract",
        categories=["cs.CL"],
        pdf_url="https://arxiv.org/pdf/2401.12345",
    )


def _state_with_pages(pages):
    state = create_initial_state("2401.12345")
    state["paper"] = _make_paper()
    state["parsed_pages"] = pages
    return state


def test_chunk_embed_success(monkeypatch):
    monkeypatch.setattr("arxiv_agent.nodes.chunk_embed.get_embedding_model", lambda: _FakeModel())
    monkeypatch.setattr(
        "arxiv_agent.nodes.chunk_embed.embed_texts",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    fake_collection = _FakeCollection("paper_2401.12345")
    monkeypatch.setattr(
        "arxiv_agent.nodes.chunk_embed.get_fresh_collection", lambda arxiv_id: fake_collection
    )

    pages = [ParsedPage(page_number=1, text="AAA BBB CCC DDD EEE FFF")]
    update = chunk_embed(_state_with_pages(pages))

    assert "error" not in update
    assert len(update["chunks"]) >= 1
    assert update["collection_name"] == "paper_2401.12345"
    assert fake_collection.added is not None
    assert len(fake_collection.added["ids"]) == len(update["chunks"])
    assert fake_collection.added["embeddings"] == [[0.1, 0.2]] * len(update["chunks"])


def test_chunk_embed_multiple_pages_produce_page_bounded_chunk_ids(monkeypatch):
    monkeypatch.setattr("arxiv_agent.nodes.chunk_embed.get_embedding_model", lambda: _FakeModel())
    monkeypatch.setattr(
        "arxiv_agent.nodes.chunk_embed.embed_texts", lambda texts: [[0.0] for _ in texts]
    )
    fake_collection = _FakeCollection("paper_2401.12345")
    monkeypatch.setattr(
        "arxiv_agent.nodes.chunk_embed.get_fresh_collection", lambda arxiv_id: fake_collection
    )

    pages = [
        ParsedPage(page_number=1, text="one two three"),
        ParsedPage(page_number=2, text="four five six"),
    ]
    update = chunk_embed(_state_with_pages(pages))
    pages_seen = {c.page for c in update["chunks"]}
    assert pages_seen == {1, 2}
    assert all(c.chunk_id.startswith("2401.12345#p") for c in update["chunks"])


def test_chunk_embed_no_extractable_text_sets_error(monkeypatch):
    monkeypatch.setattr("arxiv_agent.nodes.chunk_embed.get_embedding_model", lambda: _FakeModel())
    pages = [ParsedPage(page_number=1, text="   \n\n   ")]
    update = chunk_embed(_state_with_pages(pages))
    assert update["chunks"] == []
    assert "error" in update


def test_chunk_embed_model_load_failure(monkeypatch):
    def _raise():
        raise RuntimeError("no internet, can't download model")

    monkeypatch.setattr("arxiv_agent.nodes.chunk_embed.get_embedding_model", _raise)
    update = chunk_embed(_state_with_pages([ParsedPage(page_number=1, text="hello world")]))
    assert "error" in update


def test_chunk_embed_indexing_failure(monkeypatch):
    monkeypatch.setattr("arxiv_agent.nodes.chunk_embed.get_embedding_model", lambda: _FakeModel())
    monkeypatch.setattr(
        "arxiv_agent.nodes.chunk_embed.embed_texts", lambda texts: [[0.0] for _ in texts]
    )

    def _raise(arxiv_id):
        raise RuntimeError("disk full")

    monkeypatch.setattr("arxiv_agent.nodes.chunk_embed.get_fresh_collection", _raise)
    update = chunk_embed(_state_with_pages([ParsedPage(page_number=1, text="hello world")]))
    assert "error" in update


def test_route_after_chunk_embed():
    assert route_after_chunk_embed({"error": "boom"}) == "chunk_failed"
    assert route_after_chunk_embed({"error": None}) == "summarize"
