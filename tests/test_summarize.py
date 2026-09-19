from arxiv_agent.nodes.summarize import _build_full_text, route_after_summarize, summarize
from arxiv_agent.schemas import BriefingContent, PaperMetadata, ParsedPage
from arxiv_agent.state import create_initial_state


def _make_paper() -> PaperMetadata:
    return PaperMetadata(
        title="Test Paper",
        authors=["A. Author", "B. Author"],
        arxiv_id="2401.12345",
        published="2024-01-15",
        link="https://arxiv.org/abs/2401.12345",
        abstract="An abstract.",
        categories=["cs.CL"],
        pdf_url="https://arxiv.org/pdf/2401.12345",
    )


def _make_content() -> BriefingContent:
    return BriefingContent(
        why_it_matters="It matters because X.",
        problem_statement="The problem is Y.",
        method_approach=["Step one", "Step two"],
        key_results=["Result one"],
        limitations=["Limitation one"],
        follow_up_questions=["What about Z?"],
    )


def _state_with_pages(pages) -> dict:
    state = create_initial_state("2401.12345")
    state["paper"] = _make_paper()
    state["parsed_pages"] = pages
    return state


def test_build_full_text_passthrough_when_short():
    state = _state_with_pages([ParsedPage(page_number=1, text="short text")])
    assert _build_full_text(state, max_chars=150_000) == "short text"


def test_build_full_text_truncates_when_long():
    huge = "A" * 200_000
    state = _state_with_pages([ParsedPage(page_number=1, text=huge)])
    result = _build_full_text(state, max_chars=150_000)
    assert len(result) < len(huge)
    assert "omitted for length" in result
    assert result.startswith("A")
    assert result.endswith("A")


def test_build_full_text_respects_the_smaller_retry_budget():
    huge = "A" * 200_000
    state = _state_with_pages([ParsedPage(page_number=1, text=huge)])
    full_budget = _build_full_text(state, max_chars=150_000)
    retry_budget = _build_full_text(state, max_chars=18_000)
    assert len(retry_budget) < len(full_budget)
    assert len(retry_budget) <= 18_000 + len("\n\n[... middle of paper omitted for length ...]\n\n")


def test_summarize_success_injects_verified_metadata_not_llm_output(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt):
            return _make_content()

    monkeypatch.setattr("arxiv_agent.nodes.summarize.get_llm", lambda **kwargs: FakeLLM())

    state = _state_with_pages([ParsedPage(page_number=1, text="paper body text")])
    update = summarize(state)

    assert "error" not in update
    briefing = update["briefing"]
    # metadata comes from state["paper"] (verified arXiv data), not the LLM
    assert briefing.metadata == state["paper"]
    assert briefing.why_it_matters == "It matters because X."
    assert briefing.limitations == ["Limitation one"]


def test_summarize_retries_once_with_a_smaller_prompt_then_fails(monkeypatch):
    calls = []

    class FakeLLM:
        def invoke(self, prompt):
            calls.append(prompt)
            raise RuntimeError("provider down")

    monkeypatch.setattr("arxiv_agent.nodes.summarize.get_llm", lambda **kwargs: FakeLLM())

    huge_page = ParsedPage(page_number=1, text="A" * 200_000)
    update = summarize(_state_with_pages([huge_page]))

    assert "briefing" not in update
    assert "error" in update
    assert "retrying with a shorter excerpt" in update["error"]
    assert len(calls) == 2
    assert len(calls[1]) < len(calls[0])  # second attempt used the smaller budget


def test_summarize_succeeds_on_retry_after_first_attempt_fails(monkeypatch):
    calls = []

    class FakeLLM:
        def invoke(self, prompt):
            calls.append(prompt)
            if len(calls) == 1:
                raise RuntimeError("request too large")
            return _make_content()

    monkeypatch.setattr("arxiv_agent.nodes.summarize.get_llm", lambda **kwargs: FakeLLM())

    huge_page = ParsedPage(page_number=1, text="A" * 200_000)
    update = summarize(_state_with_pages([huge_page]))

    assert "error" not in update
    assert update["briefing"].why_it_matters == "It matters because X."
    assert len(calls) == 2


def test_route_after_summarize():
    assert route_after_summarize({"error": "boom"}) == "briefing_failed"
    assert route_after_summarize({"error": None}) == "await_question"


def test_executive_briefing_to_markdown_has_required_headings():
    from arxiv_agent.schemas import ExecutiveBriefing

    briefing = ExecutiveBriefing(metadata=_make_paper(), **_make_content().model_dump())
    md = briefing.to_markdown()
    for heading in [
        "## Metadata",
        "## Why This Paper Matters",
        "## Problem Statement",
        "## Method / Approach",
        "## Key Results / Claims",
        "## Limitations",
        "## Suggested Follow-Up Questions",
    ]:
        assert heading in md
    assert "2401.12345" in md
    assert "A. Author, B. Author" in md


def test_executive_briefing_to_markdown_omits_diagram_section_when_empty():
    from arxiv_agent.schemas import ExecutiveBriefing

    briefing = ExecutiveBriefing(metadata=_make_paper(), **_make_content().model_dump())
    assert briefing.architecture_diagram == ""
    assert "## Architecture Diagram" not in briefing.to_markdown()


def test_executive_briefing_to_markdown_includes_diagram_as_mermaid_fence():
    from arxiv_agent.schemas import ExecutiveBriefing

    content = _make_content().model_dump()
    content["architecture_diagram"] = "flowchart TD\n  A[Input] --> B[Output]"
    briefing = ExecutiveBriefing(metadata=_make_paper(), **content)
    md = briefing.to_markdown()
    assert "## Architecture Diagram" in md
    assert "```mermaid" in md
    assert "flowchart TD" in md
