"""Shared local ChromaDB access: one persistent client, one collection
per arXiv ID.

`hnsw:space: cosine` matches the embedding model's normalized vectors
(see embeddings.py) - Chroma's default is L2, which would silently
degrade retrieval quality for a model trained/intended for cosine
similarity.
"""

from __future__ import annotations

import chromadb

from arxiv_agent.config import CHROMA_DIR

_client = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def collection_name(arxiv_id: str) -> str:
    return f"paper_{arxiv_id}"


def get_fresh_collection(arxiv_id: str):
    """Return an empty collection for `arxiv_id`, deleting any prior
    contents first so re-running the pipeline on the same paper can't mix
    chunks from different chunking parameters or a partial prior run.
    """

    client = get_client()
    name = collection_name(arxiv_id)
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name=name)
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def get_existing_collection(arxiv_id: str):
    """Return the collection for `arxiv_id`, for the QA retrieval node.
    Raises if the paper hasn't been chunked/embedded yet.
    """

    return get_client().get_collection(name=collection_name(arxiv_id))
