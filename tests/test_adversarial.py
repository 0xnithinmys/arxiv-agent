"""Adversarial test: a question with no support in the paper must
produce an explicit refusal from the real pipeline, never a
hallucinated answer.

Skipped automatically when no LLM key is configured (see conftest.py).
"""

import pytest

from arxiv_agent.graph import build_qa_graph, checkpointer_context
from arxiv_agent.state import seed_qa_session

from .conftest import HAS_LLM_KEY

pytestmark = pytest.mark.skipif(
    not HAS_LLM_KEY, reason="requires OPENAI_API_KEY or GROQ_API_KEY"
)


def test_off_paper_question_triggers_refusal_not_hallucination(ingested_state):
    with checkpointer_context() as checkpointer:
        qa_graph = build_qa_graph(checkpointer)
        cfg = {
            "configurable": {"thread_id": f"{ingested_state['paper'].arxiv_id}-adversarial-btc"}
        }
        result = qa_graph.invoke(
            seed_qa_session(
                ingested_state,
                "What was the closing price of Bitcoin on the day this paper was published?",
            ),
            config=cfg,
        )

    assert result["answer"].is_grounded is False
    assert result["answer"].citations == []
