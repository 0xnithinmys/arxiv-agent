from datetime import datetime, timezone

import arxiv

from arxiv_agent.nodes.retrieve import (
    _to_paper_metadata,
    fetch_by_id,
    route_after_fetch_by_id,
    route_after_search,
    search_papers,
)
from arxiv_agent.state import create_initial_state


def _make_result(arxiv_id="2401.12345v1", title="Test Paper"):
    return arxiv.Result(
        entry_id=f"http://arxiv.org/abs/{arxiv_id}",
        updated=datetime(2024, 1, 20, tzinfo=timezone.utc),
        published=datetime(2024, 1, 15, tzinfo=timezone.utc),
        title=title,
        authors=[arxiv.Result.Author("Jane Doe"), arxiv.Result.Author("John Smith")],
        summary="A paper about testing.",
        primary_category="cs.CL",
        categories=["cs.CL", "cs.LG"],
    )


def test_to_paper_metadata_maps_fields():
    meta = _to_paper_metadata(_make_result())
    assert meta.arxiv_id == "2401.12345v1"
    assert meta.title == "Test Paper"
    assert meta.authors == ["Jane Doe", "John Smith"]
    assert meta.published == "2024-01-15"
    assert meta.link == "https://arxiv.org/abs/2401.12345v1"
    assert meta.abstract == "A paper about testing."
    assert meta.categories == ["cs.CL", "cs.LG"]


def test_fetch_by_id_success(monkeypatch):
    monkeypatch.setattr(
        "arxiv_agent.nodes.retrieve._client.results",
        lambda search: iter([_make_result()]),
    )
    state = create_initial_state("2401.12345")
    update = fetch_by_id(state)
    assert update["paper"].arxiv_id == "2401.12345v1"
    assert update["candidate_papers"] == [update["paper"]]
    assert "error" not in update


def test_fetch_by_id_not_found(monkeypatch):
    monkeypatch.setattr(
        "arxiv_agent.nodes.retrieve._client.results", lambda search: iter([])
    )
    state = create_initial_state("9999.99999")
    update = fetch_by_id(state)
    assert "9999.99999" in update["error"]


def test_fetch_by_id_network_error(monkeypatch):
    def _raise(search):
        raise arxiv.HTTPError("http://export.arxiv.org/api/query", 1, 500)

    monkeypatch.setattr("arxiv_agent.nodes.retrieve._client.results", _raise)
    state = create_initial_state("2401.12345")
    update = fetch_by_id(state)
    assert "error" in update


def test_search_papers_single_result_sets_paper(monkeypatch):
    monkeypatch.setattr(
        "arxiv_agent.nodes.retrieve._client.results",
        lambda search: iter([_make_result()]),
    )
    state = create_initial_state("a very specific topic")
    update = search_papers(state)
    assert len(update["candidate_papers"]) == 1
    assert update["paper"] == update["candidate_papers"][0]


def test_search_papers_many_results_no_auto_select(monkeypatch):
    results = [_make_result(f"240{i}.1234{i}") for i in range(3)]
    monkeypatch.setattr(
        "arxiv_agent.nodes.retrieve._client.results", lambda search: iter(results)
    )
    state = create_initial_state("broad topic")
    update = search_papers(state)
    assert len(update["candidate_papers"]) == 3
    assert "paper" not in update


def test_search_papers_zero_results(monkeypatch):
    monkeypatch.setattr(
        "arxiv_agent.nodes.retrieve._client.results", lambda search: iter([])
    )
    state = create_initial_state("nonexistent gibberish topic xyzzy")
    update = search_papers(state)
    assert update["candidate_papers"] == []
    # must set error: route_after_search sends this straight to END, and
    # without an error set here, the CLI would try to display a briefing
    # that was never generated
    assert "error" in update
    assert "xyzzy" in update["error"]


def test_route_after_fetch_by_id():
    assert route_after_fetch_by_id({"error": "bad id"}) == "no_results"
    assert route_after_fetch_by_id({"error": None}) == "fetch_parse"


def test_route_after_search():
    assert route_after_search({"candidate_papers": []}) == "no_results"
    assert route_after_search({"candidate_papers": [object()]}) == "fetch_parse"
    assert route_after_search({"candidate_papers": [object(), object()]}) == "select_paper"
