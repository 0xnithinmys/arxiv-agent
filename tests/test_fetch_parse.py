import requests

from arxiv_agent.nodes.fetch_parse import (
    _download_with_retry,
    fetch_parse,
    route_after_fetch_parse,
)
from arxiv_agent.schemas import PaperMetadata
from arxiv_agent.state import create_initial_state


def _make_paper() -> PaperMetadata:
    return PaperMetadata(
        title="Test Paper",
        authors=["A. Author"],
        arxiv_id="2401.12345",
        published="2024-01-15",
        link="https://arxiv.org/abs/2401.12345",
        abstract="An abstract.",
        categories=["cs.CL"],
        pdf_url="https://arxiv.org/pdf/2401.12345",
    )


def _state_with_paper():
    state = create_initial_state("2401.12345")
    state["paper"] = _make_paper()
    return state


def test_download_with_retry_succeeds_first_try(monkeypatch, tmp_path):
    calls = []

    def fake_download(url, dest):
        calls.append(url)

    monkeypatch.setattr("arxiv_agent.nodes.fetch_parse._download_pdf", fake_download)
    result = _download_with_retry("http://x/pdf", tmp_path / "x.pdf")
    assert result is None
    assert len(calls) == 1


def test_download_with_retry_succeeds_second_try(monkeypatch, tmp_path):
    attempts = {"n": 0}

    def fake_download(url, dest):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise requests.exceptions.ConnectionError("timeout")

    monkeypatch.setattr("arxiv_agent.nodes.fetch_parse._download_pdf", fake_download)
    monkeypatch.setattr("arxiv_agent.nodes.fetch_parse.time.sleep", lambda s: None)
    result = _download_with_retry("http://x/pdf", tmp_path / "x.pdf")
    assert result is None
    assert attempts["n"] == 2


def test_download_with_retry_fails_both_tries(monkeypatch, tmp_path):
    def fake_download(url, dest):
        raise requests.exceptions.ConnectionError("still down")

    monkeypatch.setattr("arxiv_agent.nodes.fetch_parse._download_pdf", fake_download)
    monkeypatch.setattr("arxiv_agent.nodes.fetch_parse.time.sleep", lambda s: None)
    result = _download_with_retry("http://x/pdf", tmp_path / "x.pdf")
    assert isinstance(result, requests.exceptions.ConnectionError)


def test_fetch_parse_success(monkeypatch):
    monkeypatch.setattr(
        "arxiv_agent.nodes.fetch_parse._download_with_retry", lambda url, dest: None
    )
    pages = [
        {"metadata": {"page_number": 1}, "text": "Page one text. " * 50},
        {"metadata": {"page_number": 2}, "text": "Page two text. " * 50},
    ]
    monkeypatch.setattr(
        "arxiv_agent.nodes.fetch_parse.pymupdf4llm.to_markdown",
        lambda path, page_chunks: pages,
    )

    update = fetch_parse(_state_with_paper())
    assert update["parse_failed"] is False
    assert len(update["parsed_pages"]) == 2
    assert update["parsed_pages"][0].page_number == 1
    assert "error" not in update


def test_fetch_parse_download_failure(monkeypatch):
    monkeypatch.setattr(
        "arxiv_agent.nodes.fetch_parse._download_with_retry",
        lambda url, dest: ConnectionError("boom"),
    )
    update = fetch_parse(_state_with_paper())
    assert update["parse_failed"] is True
    assert "error" in update


def test_fetch_parse_scanned_pdf_too_little_text(monkeypatch):
    monkeypatch.setattr(
        "arxiv_agent.nodes.fetch_parse._download_with_retry", lambda url, dest: None
    )
    pages = [{"metadata": {"page_number": 1}, "text": "short"}]
    monkeypatch.setattr(
        "arxiv_agent.nodes.fetch_parse.pymupdf4llm.to_markdown",
        lambda path, page_chunks: pages,
    )
    update = fetch_parse(_state_with_paper())
    assert update["parse_failed"] is True
    assert "scanned" in update["error"]


def test_fetch_parse_exception_during_parse(monkeypatch):
    monkeypatch.setattr(
        "arxiv_agent.nodes.fetch_parse._download_with_retry", lambda url, dest: None
    )

    def _raise(path, page_chunks):
        raise RuntimeError("corrupt PDF")

    monkeypatch.setattr("arxiv_agent.nodes.fetch_parse.pymupdf4llm.to_markdown", _raise)
    update = fetch_parse(_state_with_paper())
    assert update["parse_failed"] is True
    assert "error" in update


def test_route_after_fetch_parse():
    assert route_after_fetch_parse({"parse_failed": True}) == "parse_failed"
    assert route_after_fetch_parse({"parse_failed": False}) == "chunk_embed"
