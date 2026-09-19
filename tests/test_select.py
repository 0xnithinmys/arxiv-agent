from arxiv_agent.nodes.select import _format_candidates, select_paper
from arxiv_agent.schemas import PaperMetadata, PaperSelection
from arxiv_agent.state import create_initial_state


def _make_paper(i: int, title: str | None = None, n_authors: int = 2) -> PaperMetadata:
    return PaperMetadata(
        title=title or f"Paper {i}",
        authors=[f"Author {j}" for j in range(n_authors)],
        arxiv_id=f"240{i}.1234{i}",
        published="2024-01-15",
        link=f"https://arxiv.org/abs/240{i}.1234{i}",
        abstract=f"Abstract text for paper {i}. " * 20,
        categories=["cs.CL"],
    )


def test_format_candidates_truncates_long_abstract_and_lists_et_al():
    papers = [_make_paper(0, n_authors=5)]
    formatted = _format_candidates(papers)
    assert "[0] Title: Paper 0" in formatted
    assert "et al." in formatted
    assert formatted.count("...") >= 1


def test_select_paper_uses_llm_index(monkeypatch):
    candidates = [_make_paper(0), _make_paper(1), _make_paper(2)]
    state = create_initial_state("some topic")
    state["candidate_papers"] = candidates

    class FakeLLM:
        def invoke(self, prompt):
            return PaperSelection(index=1, justification="best topical match")

    monkeypatch.setattr("arxiv_agent.nodes.select.get_llm", lambda **kwargs: FakeLLM())

    update = select_paper(state)
    assert update["paper"] == candidates[1]


def test_select_paper_falls_back_on_out_of_range_index(monkeypatch):
    candidates = [_make_paper(0), _make_paper(1)]
    state = create_initial_state("some topic")
    state["candidate_papers"] = candidates

    class FakeLLM:
        def invoke(self, prompt):
            return PaperSelection(index=99, justification="oops")

    monkeypatch.setattr("arxiv_agent.nodes.select.get_llm", lambda **kwargs: FakeLLM())

    update = select_paper(state)
    assert update["paper"] == candidates[0]


def test_select_paper_falls_back_on_llm_exception(monkeypatch):
    candidates = [_make_paper(0), _make_paper(1)]
    state = create_initial_state("some topic")
    state["candidate_papers"] = candidates

    class FakeLLM:
        def invoke(self, prompt):
            raise RuntimeError("provider is down")

    monkeypatch.setattr("arxiv_agent.nodes.select.get_llm", lambda **kwargs: FakeLLM())

    update = select_paper(state)
    assert update["paper"] == candidates[0]
