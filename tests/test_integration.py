"""Integration test: a real arXiv ID -> a real briefing -> a real
grounded QA answer, through the actual compiled graphs - not mocks.

Skipped automatically when no LLM key is configured (see conftest.py).
"""

import pytest

from arxiv_agent.graph import build_qa_graph, checkpointer_context
from arxiv_agent.state import seed_qa_session

from .conftest import HAS_LLM_KEY

pytestmark = pytest.mark.skipif(
    not HAS_LLM_KEY, reason="requires OPENAI_API_KEY or GROQ_API_KEY"
)


def test_ingestion_produces_a_valid_briefing(ingested_state):
    assert ingested_state["error"] is None
    assert ingested_state["paper"].title == "Attention Is All You Need"
    assert len(ingested_state["chunks"]) > 0

    briefing = ingested_state["briefing"]
    assert briefing is not None
    assert briefing.metadata.arxiv_id.startswith("1706.03762")
    assert len(briefing.method_approach) >= 1
    assert len(briefing.key_results) >= 1
    assert len(briefing.limitations) >= 1  # schema enforces this too; confirm it end to end


def test_qa_answers_a_real_question_with_a_valid_citation(ingested_state):
    with checkpointer_context() as checkpointer:
        qa_graph = build_qa_graph(checkpointer)
        cfg = {
            "configurable": {"thread_id": f"{ingested_state['paper'].arxiv_id}-integration-qa"}
        }
        result = qa_graph.invoke(
            seed_qa_session(ingested_state, "How many attention heads does the base model use?"),
            config=cfg,
        )

    assert result["answer"].is_grounded is True
    assert len(result["answer"].citations) > 0
    assert "8" in result["answer"].answer
    assert len(result["conversation_history"]) == 1
