from arxiv_agent.schemas import ExecutiveBriefing, PaperMetadata
from arxiv_agent.state import create_initial_state, create_qa_turn_input, seed_qa_session


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


def test_create_qa_turn_input_resets_turn_scoped_fields():
    update = create_qa_turn_input("a new question")
    assert update["question"] == "a new question"
    assert update["regeneration_count"] == 0
    assert update["retrieved_chunks"] == []
    assert update["answer"] is None
    assert update["error"] is None
    # session-scoped fields must NOT appear - the checkpointer supplies them
    assert "paper" not in update
    assert "conversation_history" not in update


def test_seed_qa_session_carries_over_session_fields_and_resets_turn_fields():
    paper = _make_paper()
    ingestion_state = create_initial_state("2401.12345")
    ingestion_state["paper"] = paper
    ingestion_state["conversation_history"] = []
    ingestion_state["briefing"] = ExecutiveBriefing(
        metadata=paper,
        why_it_matters="x",
        problem_statement="x",
        method_approach=["x"],
        key_results=["x"],
        limitations=["x"],
    )
    # simulate a stale regeneration_count from earlier in the ingestion run
    ingestion_state["regeneration_count"] = 1

    seeded = seed_qa_session(ingestion_state, "first question")
    assert seeded["paper"] == paper
    assert seeded["briefing"] is not None
    assert seeded["conversation_history"] == []
    assert seeded["question"] == "first question"
    assert seeded["regeneration_count"] == 0  # reset, not carried over from ingestion
