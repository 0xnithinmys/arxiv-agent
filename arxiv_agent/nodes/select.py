"""Selection / Ranking node: pick the best-matching candidate for a topic
search that returned multiple results.

Only reached via `route_after_search`'s "many" branch - the 0/1 cases are
already resolved before this node exists, so no routing logic is needed
here: this node always ends by handing off to Fetch & Parse.
"""

from __future__ import annotations

import logging

from arxiv_agent.llm import get_llm
from arxiv_agent.schemas import PaperMetadata, PaperSelection
from arxiv_agent.state import GraphState

logger = logging.getLogger(__name__)

_ABSTRACT_PREVIEW_CHARS = 500


def _format_candidates(papers: list[PaperMetadata]) -> str:
    blocks = []
    for i, paper in enumerate(papers):
        abstract = paper.abstract[:_ABSTRACT_PREVIEW_CHARS]
        if len(paper.abstract) > _ABSTRACT_PREVIEW_CHARS:
            abstract += "..."
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += ", et al."
        blocks.append(
            f"[{i}] Title: {paper.title}\n"
            f"    Authors: {authors}\n"
            f"    Published: {paper.published}\n"
            f"    Categories: {', '.join(paper.categories)}\n"
            f"    Abstract: {abstract}"
        )
    return "\n\n".join(blocks)


def select_paper(state: GraphState) -> dict:
    """Pick the candidate that best matches the user's topic.

    Falls back to `candidates[0]` - the top-ranked result, since arXiv's
    Search already sorts by relevance - if the LLM call fails outright or
    returns an out-of-range index, so a flaky LLM never blocks the
    pipeline. This is a deliberately broad except: any failure mode here
    should degrade to the same safe fallback, not just specific ones.
    """

    candidates = state["candidate_papers"]
    topic = state["raw_input"]

    try:
        llm = get_llm(temperature=0.0, output_schema=PaperSelection)
        selection = llm.invoke(
            "You are choosing which arXiv paper best matches a researcher's topic.\n\n"
            f"Topic: {topic}\n\n"
            f"Candidates:\n{_format_candidates(candidates)}\n\n"
            "Pick the single best-matching candidate by its index."
        )
        if not 0 <= selection.index < len(candidates):
            raise ValueError(
                f"index {selection.index} out of range for {len(candidates)} candidates"
            )
        logger.info("Selected candidate %d: %s", selection.index, selection.justification)
        return {"paper": candidates[selection.index]}
    except Exception as exc:
        logger.warning("Paper selection failed (%s); falling back to top result.", exc)
        return {"paper": candidates[0]}
