"""Shared fixtures for integration/adversarial tests that hit the real
pipeline (live arXiv API, real embeddings, and a real LLM).

These tests are skipped automatically when no LLM key is configured, so
the default `pytest` run needs zero paid keys - matching the
assessment's own "no paid API key required to run the submission." When
a key *is* present, they run for real: that's what actually proves the
pipeline works end to end, not just its individual pieces in isolation.
"""

import pytest

from arxiv_agent.config import GROQ_API_KEY, OPENAI_API_KEY
from arxiv_agent.graph import build_ingestion_graph
from arxiv_agent.state import create_initial_state

HAS_LLM_KEY = bool(OPENAI_API_KEY or GROQ_API_KEY)

# Attention Is All You Need: small (15 pages), stable, well-known - a
# good fixed target for tests that need a real paper end to end.
TEST_ARXIV_ID = "1706.03762"


@pytest.fixture(scope="session")
def ingested_state():
    """Run the real ingestion graph once per test session; every
    integration/adversarial test reuses this instead of re-running the
    live arXiv fetch + parse + chunk + embed + summarize pipeline.
    """

    graph = build_ingestion_graph()
    return graph.invoke(create_initial_state(TEST_ARXIV_ID))
