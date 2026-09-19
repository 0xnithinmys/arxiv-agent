"""Assembles the two compiled LangGraph StateGraphs.

**Ingestion graph** (query understanding through the executive briefing)
is a one-shot flow: nothing about it needs to be resumed mid-way, so it
runs without a meaningful thread_id. Its final state (still available in
memory as the return value of `.invoke()`) is what seeds the QA graph.

**QA graph** (one grounded question-answer turn) is checkpointed under
`thread_id = paper.arxiv_id` via `SqliteSaver` - a file-backed store, not
`MemorySaver`, which loses everything on process restart. This project
assumes the CLI may run summarize and QA as separate invocations (see
README), so only a file-backed checkpointer keeps the paper/chunks/
briefing/conversation-history available if the process restarts between
them. The first QA call for a paper must be seeded with
`state.seed_qa_session(...)` since the checkpointer has never seen that
thread_id before; every later call - same process or a fresh one - only
needs `state.create_qa_turn_input(...)`, because the checkpointer now
supplies the session fields itself.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from typing import Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from arxiv_agent.config import CHECKPOINT_DB_PATH
from arxiv_agent.nodes.chunk_embed import chunk_embed, route_after_chunk_embed
from arxiv_agent.nodes.fetch_parse import fetch_parse, route_after_fetch_parse
from arxiv_agent.nodes.qa import (
    answer_question,
    prepare_regeneration,
    record_turn,
    refuse,
    retrieve_chunks,
    route_after_answer,
    route_after_retrieve,
    route_after_verify,
    verify_answer,
)
from arxiv_agent.nodes.retrieve import (
    fetch_by_id,
    route_after_fetch_by_id,
    route_after_search,
    search_papers,
)
from arxiv_agent.nodes.select import select_paper
from arxiv_agent.nodes.summarize import route_after_summarize, summarize
from arxiv_agent.nodes.understand_query import route_after_understanding, understand_query
from arxiv_agent.state import GraphState


# GraphState carries Pydantic models (PaperMetadata, ParsedPage, Chunk,
# ExecutiveBriefing, QAAnswer) that SqliteSaver's default serializer will
# still checkpoint today, but only with a "this will be blocked in a
# future version" warning on every read - it isn't in the serializer's
# built-in safe-type list. Explicitly allowlisting them (rather than
# leaving the default permissive-with-warning behavior) is what keeps
# this working once a future langgraph defaults to strict mode.
_ALLOWED_MSGPACK_MODULES = [
    ("arxiv_agent.schemas", "PaperMetadata"),
    ("arxiv_agent.schemas", "ParsedPage"),
    ("arxiv_agent.schemas", "Chunk"),
    ("arxiv_agent.schemas", "ExecutiveBriefing"),
    ("arxiv_agent.schemas", "QAAnswer"),
]


def _build_checkpointer(conn: sqlite3.Connection) -> SqliteSaver:
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    return SqliteSaver(conn, serde=serde)


@contextmanager
def checkpointer_context() -> Iterator[SqliteSaver]:
    """Open the shared SqliteSaver connection for a short-lived caller.

    One context per CLI session, backing both graphs -
    `with checkpointer_context() as checkpointer:` then pass it to both
    `build_ingestion_graph` and `build_qa_graph`. Mirrors
    `SqliteSaver.from_conn_string`'s own connection setup (same
    `check_same_thread=False` rationale - sqlite3 connections are
    thread-bound by default) since that factory has no way to pass a
    custom `serde` through it.
    """

    with closing(
        sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    ) as conn:
        yield _build_checkpointer(conn)


def build_ingestion_graph(checkpointer=None) -> CompiledStateGraph:
    """Query Understanding -> arXiv Retrieval -> Selection -> Fetch & Parse
    -> Chunk & Embed -> Executive Briefing, per the brief's own diagram.
    """

    graph = StateGraph(GraphState)

    graph.add_node("understand_query", understand_query)
    graph.add_node("fetch_by_id", fetch_by_id)
    graph.add_node("search_papers", search_papers)
    graph.add_node("select_paper", select_paper)
    graph.add_node("fetch_parse", fetch_parse)
    graph.add_node("chunk_embed", chunk_embed)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("understand_query")

    graph.add_conditional_edges(
        "understand_query",
        route_after_understanding,
        {
            "validation_error": END,
            "fetch_by_id": "fetch_by_id",
            "search_papers": "search_papers",
        },
    )
    graph.add_conditional_edges(
        "fetch_by_id",
        route_after_fetch_by_id,
        {"no_results": END, "fetch_parse": "fetch_parse"},
    )
    graph.add_conditional_edges(
        "search_papers",
        route_after_search,
        {"no_results": END, "fetch_parse": "fetch_parse", "select_paper": "select_paper"},
    )
    graph.add_edge("select_paper", "fetch_parse")
    graph.add_conditional_edges(
        "fetch_parse",
        route_after_fetch_parse,
        {"parse_failed": END, "chunk_embed": "chunk_embed"},
    )
    graph.add_conditional_edges(
        "chunk_embed",
        route_after_chunk_embed,
        {"chunk_failed": END, "summarize": "summarize"},
    )
    graph.add_conditional_edges(
        "summarize",
        route_after_summarize,
        # both outcomes end the ingestion graph; the CLI distinguishes them
        # by checking final_state["error"] vs. final_state["briefing"]
        {"briefing_failed": END, "await_question": END},
    )

    return graph.compile(checkpointer=checkpointer)


def build_qa_graph(checkpointer) -> CompiledStateGraph:
    """retrieve_chunks -> answer_question -> verify_answer, looping back
    through prepare_regeneration for exactly one retry before refuse. A
    retrieval failure or a generation failure both skip straight to
    refuse rather than running the next node on incomplete state - every
    path converges on record_turn.
    """

    graph = StateGraph(GraphState)

    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("answer_question", answer_question)
    graph.add_node("verify_answer", verify_answer)
    graph.add_node("prepare_regeneration", prepare_regeneration)
    graph.add_node("refuse", refuse)
    graph.add_node("record_turn", record_turn)

    graph.set_entry_point("retrieve_chunks")
    graph.add_conditional_edges(
        "retrieve_chunks",
        route_after_retrieve,
        {"retrieval_failed": "refuse", "answer_question": "answer_question"},
    )
    graph.add_conditional_edges(
        "answer_question",
        route_after_answer,
        {"answer_failed": "refuse", "verify_answer": "verify_answer"},
    )
    graph.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {
            "return_answer": "record_turn",
            "regenerate": "prepare_regeneration",
            "refuse": "refuse",
        },
    )
    graph.add_edge("prepare_regeneration", "answer_question")
    graph.add_edge("refuse", "record_turn")
    graph.add_edge("record_turn", END)

    return graph.compile(checkpointer=checkpointer)
