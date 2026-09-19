"""Grounded QA loop: retrieve -> answer -> code-verify -> regenerate-once-
or-refuse.

The code-verification step is the key differentiator called out in the
project's own evaluation of the source documents: instead of trusting the
LLM's self-reported `is_grounded` flag (or a fragile "NOT_SUPPORTED"
string match), every cited chunk_id is checked in code against the
chunks that were actually retrieved this turn. On a failed verification,
the loop regenerates exactly once with a stricter prompt before refusing
outright - bounded so a stubbornly ungrounded question can't loop forever.
"""

from __future__ import annotations

import logging

from arxiv_agent.chroma_store import get_existing_collection
from arxiv_agent.embeddings import embed_texts
from arxiv_agent.llm import get_llm
from arxiv_agent.schemas import Chunk, PaperMetadata, QAAnswer
from arxiv_agent.state import GraphState

logger = logging.getLogger(__name__)

_TOP_K = 4
_MAX_REGENERATIONS = 1
_REFUSAL_MESSAGE = "I couldn't find support for that in the paper."


def _metadata_chunk_id(arxiv_id: str) -> str:
    return f"{arxiv_id}#metadata"


def _metadata_chunk(paper: PaperMetadata) -> Chunk:
    """A synthetic, always-available chunk carrying the paper's own
    verified metadata (from the arXiv API, not vector search).

    Without this, a question like "who are the authors" is answered
    purely by whatever the vector search happens to retrieve - and a
    paper's references section is often *more* semantically similar to
    an "authors" query than its own title-page chunk is (a references
    list is dense with "Name, Name, ... Title, Venue, Year" patterns
    repeated dozens of times), so the model can end up answering with
    the authors of papers *cited by* this one instead of this paper's
    own authors. This chunk is included in every turn's retrieved set
    specifically so metadata questions have a correct, citable source
    regardless of what the vector search ranks highest.
    """

    return Chunk(
        chunk_id=_metadata_chunk_id(paper.arxiv_id),
        doc_id=paper.arxiv_id,
        section="Metadata",
        text=(
            f"Title: {paper.title}\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Published: {paper.published}\n"
            f"arXiv ID: {paper.arxiv_id}\n"
            f"Categories: {', '.join(paper.categories)}"
        ),
    )


_ANSWER_SYSTEM_PROMPT_TEMPLATE = (
    "You are answering a question about the paper \"{title}\" (arXiv:{arxiv_id}), "
    "using ONLY the excerpts provided below - each is labeled with its chunk_id. "
    "\"The paper\" always means this specific paper, never a different paper that "
    "is merely cited, discussed, or listed in its references section. Cite the "
    "chunk_id of every excerpt you rely on in `citations`. If the excerpts "
    "do not contain enough information to answer the question about THIS paper, "
    "set is_grounded to false and say so plainly in `answer` rather than "
    "guessing or using outside knowledge."
)

_STRICT_RETRY_SUFFIX = (
    "\n\nIMPORTANT: A previous attempt to answer this question was rejected "
    "because it wasn't properly grounded (no citation, a citation to an "
    "excerpt that doesn't exist, or a grounded claim the evidence didn't "
    "support). This time, only cite chunk_ids that literally appear in the "
    "excerpts above, and if you cannot support an answer from them, set "
    "is_grounded to false and say so instead of guessing."
)


def _format_chunks_for_prompt(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)


def retrieve_chunks(state: GraphState) -> dict:
    """Embed the question and pull the top-k most relevant chunks from
    this paper's Chroma collection.
    """

    question = state["question"]
    paper = state["paper"]

    try:
        query_embedding = embed_texts([question])[0]
        collection = get_existing_collection(paper.arxiv_id)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=_TOP_K,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.warning("Chunk retrieval failed for %s: %s", paper.arxiv_id, exc)
        return {"error": f"Could not retrieve context for the question: {exc}"}

    retrieved = [
        Chunk(chunk_id=chunk_id, doc_id=meta["doc_id"], section=meta["section"],
              page=meta["page"], text=doc)
        for chunk_id, doc, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        )
    ]
    retrieved.append(_metadata_chunk(paper))
    return {"retrieved_chunks": retrieved}


def route_after_retrieve(state: GraphState) -> str:
    """Conditional-edge router: skip straight to refusal on a retrieval
    failure rather than letting `answer_question` run against zero
    chunks - without this, a Chroma/embedding failure would silently
    disappear (the retrieval error gets overwritten by whatever the LLM
    says about an empty excerpt list) instead of surfacing to the user,
    and would waste an LLM call that's guaranteed to be unhelpful.
    """

    return "retrieval_failed" if state["error"] else "answer_question"


def answer_question(state: GraphState) -> dict:
    """Generate a structured, citation-aware answer from the retrieved chunks.

    Uses the stricter retry prompt whenever `regeneration_count > 0` - i.e.
    this is a regeneration after a failed verification, not the first attempt.
    """

    question = state["question"]
    retrieved = state["retrieved_chunks"]
    paper = state["paper"]
    is_retry = state["regeneration_count"] > 0

    system_prompt = _ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        title=paper.title, arxiv_id=paper.arxiv_id
    )
    prompt = (
        f"{system_prompt}\n\n"
        f"Question: {question}\n\n"
        f"Excerpts:\n{_format_chunks_for_prompt(retrieved)}"
    )
    if is_retry:
        prompt += _STRICT_RETRY_SUFFIX

    try:
        llm = get_llm(temperature=0.0, output_schema=QAAnswer)
        answer = llm.invoke(prompt)
    except Exception as exc:
        logger.warning("QA answer generation failed: %s", exc)
        return {"error": f"Could not generate an answer: {exc}"}

    return {"answer": answer}


def route_after_answer(state: GraphState) -> str:
    """Conditional-edge router: skip verify_answer on a generation
    failure. Without this, a failed `answer_question` (its except branch
    sets `error` but leaves `answer` at the None a fresh turn starts
    with) would crash `verify_answer` on `state["answer"].citations` -
    reachable on both the first attempt and a regeneration retry, since
    both loop back through this same node.
    """

    return "answer_failed" if state["error"] else "verify_answer"


def _metadata_chunk_supports_answer(paper: PaperMetadata, answer_text: str) -> bool:
    """Whether an answer citing *only* the metadata chunk actually draws on
    what that chunk says, rather than citing it as a rubber stamp.

    The metadata chunk is injected into every turn's retrieved set
    regardless of the question (see `_metadata_chunk`), so its chunk_id is
    trivially always "in the retrieved set" - `verify_answer`'s membership
    check alone can't catch a model citing it to justify a claim about
    something else entirely (confirmed live: a question about an unrelated
    topic got answered, self-reported as grounded, and cited only
    `#metadata` - membership passed even though the metadata chunk has
    nothing to do with the claim). This requires the answer text to
    actually contain one of the fields that chunk holds - the only claims
    it can honestly support.
    """

    haystack = answer_text.lower()
    fields = [paper.title, paper.published, paper.arxiv_id, *paper.authors, *paper.categories]
    return any(field and field.lower() in haystack for field in fields)


def verify_answer(state: GraphState) -> dict:
    """Code-verify the LLM's citations against what was actually retrieved.

    `is_grounded` is recomputed here, not trusted from the model: it
    requires a non-empty citation list, every cited chunk_id to be a
    subset of the retrieved set (no citing a chunk the model never saw),
    AND the model's own self-reported flag to agree. When the *only*
    citation is the always-present metadata chunk, membership alone would
    pass for any question regardless of topic, so that case additionally
    requires the answer text to actually draw on the metadata chunk's own
    fields (see `_metadata_chunk_supports_answer`).
    """

    answer = state["answer"]
    paper = state["paper"]
    retrieved_ids = {c.chunk_id for c in state["retrieved_chunks"]}

    cited_ids = set(answer.citations)
    citations_are_valid = bool(cited_ids) and cited_ids.issubset(retrieved_ids)

    if citations_are_valid and cited_ids == {_metadata_chunk_id(paper.arxiv_id)}:
        citations_are_valid = _metadata_chunk_supports_answer(paper, answer.answer)

    verified_grounded = answer.is_grounded and citations_are_valid

    verified_answer = answer.model_copy(update={"is_grounded": verified_grounded})
    return {"answer": verified_answer}


def route_after_verify(state: GraphState) -> str:
    """Conditional-edge router: return the grounded answer, loop back for
    exactly one regeneration, or refuse once that retry is exhausted.
    """

    if state["answer"].is_grounded:
        return "return_answer"
    if state["regeneration_count"] < _MAX_REGENERATIONS:
        return "regenerate"
    return "refuse"


def prepare_regeneration(state: GraphState) -> dict:
    """Bump the regeneration counter before looping back into
    `answer_question` - routers can't mutate state themselves, so this
    tiny node is what makes the retry visible to both the router (to stop
    after one attempt) and `answer_question` (to switch to the stricter
    prompt).
    """

    return {"regeneration_count": state["regeneration_count"] + 1}


def refuse(state: GraphState) -> dict:
    """Final refusal once regeneration is exhausted and the answer still
    isn't grounded - explicit and honest rather than a plausible-sounding
    but unsupported answer.
    """

    return {
        "answer": QAAnswer(answer=_REFUSAL_MESSAGE, citations=[], is_grounded=False)
    }


def record_turn(state: GraphState) -> dict:
    """Append the completed turn - accepted or refused, both pass through
    here - to conversation history. Nothing else in the QA loop writes to
    conversation_history, so this is the one place a turn becomes part of
    the persisted session.
    """

    turn = {"question": state["question"], "answer": state["answer"]}
    return {"conversation_history": state["conversation_history"] + [turn]}
