"""The shared graph state.

Plain TypedDict, not a Pydantic BaseModel: current LangGraph guidance treats
TypedDict as the default state schema and notes that a full Pydantic state
is less performant. Pydantic is used instead for the node *outputs* that
need real validation (see schemas.py).
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

from arxiv_agent.schemas import Chunk, ExecutiveBriefing, PaperMetadata, ParsedPage, QAAnswer


class QATurn(TypedDict):
    """One exchange in the QA conversation history."""

    question: str
    answer: QAAnswer


class GraphState(TypedDict):
    # --- input / query understanding ---
    raw_input: str
    input_type: Optional[Literal["arxiv_id", "topic"]]

    # --- arXiv retrieval / selection ---
    candidate_papers: list[PaperMetadata]
    paper: Optional[PaperMetadata]

    # --- fetch & parse ---
    parsed_pages: list[ParsedPage]
    parse_failed: bool

    # --- chunk & embed ---
    chunks: list[Chunk]
    collection_name: Optional[str]

    # --- executive briefing ---
    briefing: Optional[ExecutiveBriefing]

    # --- QA loop ---
    question: Optional[str]
    retrieved_chunks: list[Chunk]
    answer: Optional[QAAnswer]
    regeneration_count: int
    conversation_history: list[QATurn]

    # --- error handling ---
    error: Optional[str]


def create_initial_state(raw_input: str) -> GraphState:
    """Build a fresh, fully-keyed state for a new run.

    LangGraph merges each node's returned partial dict into this state, but
    every key must exist from the start so early nodes can read defaults
    (e.g. `regeneration_count == 0`) before anything has written to them.
    """

    return GraphState(
        raw_input=raw_input,
        input_type=None,
        candidate_papers=[],
        paper=None,
        parsed_pages=[],
        parse_failed=False,
        chunks=[],
        collection_name=None,
        briefing=None,
        question=None,
        retrieved_chunks=[],
        answer=None,
        regeneration_count=0,
        conversation_history=[],
        error=None,
    )


def create_qa_turn_input(question: str) -> dict:
    """Partial state update to start one QA-graph invocation.

    Turn-scoped fields are explicitly reset here - most importantly
    `regeneration_count`: without this reset, a brand new question would
    silently inherit a *previous* question's leftover count from the
    checkpoint (both questions share the same thread_id) and could start
    already treated as having used its one retry. Session-scoped fields
    (paper, chunks, briefing, conversation_history, ...) are left out on
    purpose - the checkpointer supplies them from prior turns under the
    same thread_id.
    """

    return {
        "question": question,
        "regeneration_count": 0,
        "retrieved_chunks": [],
        "answer": None,
        "error": None,
    }


def seed_qa_session(ingestion_state: GraphState, question: str) -> dict:
    """Full state update to start the *first* QA-graph invocation for a
    paper, right after the ingestion graph produced it.

    The QA graph is checkpointed under `thread_id = paper.arxiv_id`, a
    thread the checkpointer has never seen before this call - so unlike
    later turns (`create_qa_turn_input` alone is enough once the
    checkpoint exists), this first call must carry over the session
    fields the ingestion graph computed in memory, or `retrieve_chunks`
    would have no `paper` to look up a collection for.
    """

    return {
        "paper": ingestion_state["paper"],
        "chunks": ingestion_state["chunks"],
        "collection_name": ingestion_state["collection_name"],
        "parsed_pages": ingestion_state["parsed_pages"],
        "briefing": ingestion_state["briefing"],
        "conversation_history": ingestion_state["conversation_history"],
        **create_qa_turn_input(question),
    }
