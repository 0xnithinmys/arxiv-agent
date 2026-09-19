"""Chunk & Embed node: split parsed pages into token-budgeted chunks,
embed them locally, and store them in a per-paper Chroma collection.

Chunks are page-bounded (never span two pages) so each chunk's `page`
metadata is always exact - a deliberate simplification over merging
short pages together, matching the brief's own "page-level chunks as a
defensible, simple default for a single paper."
"""

from __future__ import annotations

import logging

from arxiv_agent.chroma_store import get_fresh_collection
from arxiv_agent.chunking import chunk_text
from arxiv_agent.embeddings import EMBEDDING_MODEL_NAME, embed_texts, get_embedding_model
from arxiv_agent.schemas import Chunk
from arxiv_agent.state import GraphState

logger = logging.getLogger(__name__)

# Safely under bge-small-en-v1.5's verified 512-token max_seq_length,
# leaving room for BERT's [CLS]/[SEP] special tokens plus a safety
# margin - not the 800-1000 tokens a longer-context model could support
# (see embeddings.py for why that model was rejected).
CHUNK_TARGET_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 70


def chunk_embed(state: GraphState) -> dict:
    """Chunk every parsed page, embed all chunks in one batch, and store
    them in a fresh per-paper Chroma collection.
    """

    paper = state["paper"]

    try:
        tokenizer = get_embedding_model().tokenizer
    except Exception as exc:
        logger.warning("Failed to load the embedding model: %s", exc)
        return {
            "error": f"Could not load the local embedding model ({EMBEDDING_MODEL_NAME}): {exc}"
        }

    chunks: list[Chunk] = []
    for page in state["parsed_pages"]:
        text_chunks = chunk_text(
            page.text,
            tokenizer,
            target_tokens=CHUNK_TARGET_TOKENS,
            overlap_tokens=CHUNK_OVERLAP_TOKENS,
        )
        for i, tc in enumerate(text_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.arxiv_id}#p{page.page_number}_{i}",
                    doc_id=paper.arxiv_id,
                    section=tc.section,
                    page=page.page_number,
                    text=tc.text,
                )
            )

    if not chunks:
        return {"chunks": [], "error": f"No chunkable text was extracted for {paper.arxiv_id}."}

    try:
        embeddings = embed_texts([c.text for c in chunks])
        collection = get_fresh_collection(paper.arxiv_id)
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc_id": c.doc_id, "section": c.section, "page": c.page} for c in chunks
            ],
        )
    except Exception as exc:
        logger.warning("Embedding/indexing failed for %s: %s", paper.arxiv_id, exc)
        return {"error": f"Could not embed/index {paper.arxiv_id}: {exc}"}

    logger.info("Embedded %d chunks for %s", len(chunks), paper.arxiv_id)
    return {"chunks": chunks, "collection_name": collection.name}


def route_after_chunk_embed(state: GraphState) -> str:
    """Conditional-edge router: embedding failure vs. ready to summarize."""

    return "chunk_failed" if state["error"] else "summarize"
