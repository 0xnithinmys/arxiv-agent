"""arXiv Retrieval node: fetch-by-ID and topic search, both via the
official `arxiv` package - no scraping.

A single shared `arxiv.Client` enforces arXiv's real rate limit (one
request every 3 seconds, single connection - see config.py) across every
call made here: the package tracks the last request timestamp on the
client instance itself and sleeps before any request that would violate
`delay_seconds`, so reusing one client is what makes the limit apply
globally rather than resetting per call.
"""

from __future__ import annotations

import logging

import arxiv
import requests

from arxiv_agent.config import ARXIV_REQUEST_INTERVAL_SECONDS
from arxiv_agent.schemas import PaperMetadata
from arxiv_agent.state import GraphState

logger = logging.getLogger(__name__)

_MAX_TOPIC_RESULTS = 5

# Precise external-boundary exceptions: arxiv.ArxivError (HTTPError /
# UnexpectedEmptyPageError) covers arXiv-specific failures, but the
# package re-raises the raw requests exception once its own retries are
# exhausted, so both must be caught.
_ARXIV_EXCEPTIONS = (arxiv.ArxivError, requests.exceptions.RequestException)

_client = arxiv.Client(delay_seconds=ARXIV_REQUEST_INTERVAL_SECONDS)


def _to_paper_metadata(result: arxiv.Result) -> PaperMetadata:
    short_id = result.get_short_id()
    return PaperMetadata(
        title=result.title,
        authors=[author.name for author in result.authors],
        arxiv_id=short_id,
        published=result.published.date().isoformat(),
        link=f"https://arxiv.org/abs/{short_id}",
        abstract=result.summary,
        categories=list(result.categories),
        pdf_url=result.pdf_url,
    )


def fetch_by_id(state: GraphState) -> dict:
    """Look up a single paper by its arXiv ID (state["raw_input"], already
    normalized by Query Understanding). Deterministic: at most one result,
    so no selection/ranking step is needed on this path.
    """

    arxiv_id = state["raw_input"]

    try:
        results = list(_client.results(arxiv.Search(id_list=[arxiv_id])))
    except _ARXIV_EXCEPTIONS as exc:
        logger.warning("arXiv lookup failed for id=%s: %s", arxiv_id, exc)
        return {"error": f"Could not reach arXiv to fetch '{arxiv_id}'. Please try again."}

    if not results:
        return {
            "error": (
                f"'{arxiv_id}' doesn't match a paper on arXiv. Double-check the ID "
                "(e.g. 2401.12345) or paste the arxiv.org URL directly."
            )
        }

    paper = _to_paper_metadata(results[0])
    return {"paper": paper, "candidate_papers": [paper]}


def route_after_fetch_by_id(state: GraphState) -> str:
    """Conditional-edge router: invalid/unreachable ID vs. found."""

    return "no_results" if state["error"] else "fetch_parse"


def search_papers(state: GraphState) -> dict:
    """Topic search: up to `_MAX_TOPIC_RESULTS` candidates, ranked by relevance.

    When the search happens to return exactly one candidate, `paper` is set
    immediately alongside `candidate_papers` so the single-result routing
    branch can skip straight to Fetch & Parse without a separate selection
    step needing to run first.
    """

    topic = state["raw_input"]

    try:
        search = arxiv.Search(
            query=topic,
            max_results=_MAX_TOPIC_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(_client.results(search))
    except _ARXIV_EXCEPTIONS as exc:
        logger.warning("arXiv search failed for query=%r: %s", topic, exc)
        return {"error": f"Could not reach arXiv to search for '{topic}'. Please try again."}

    candidates = [_to_paper_metadata(r) for r in results]

    if not candidates:
        return {
            "candidate_papers": [],
            "error": (
                f"No arXiv papers matched '{topic}'. Try a broader or "
                "differently-worded topic, or paste a specific arXiv ID/URL."
            ),
        }

    update: dict = {"candidate_papers": candidates}
    if len(candidates) == 1:
        update["paper"] = candidates[0]
    return update


def route_after_search(state: GraphState) -> str:
    """Conditional-edge router on result count: 0 / 1 / many."""

    count = len(state["candidate_papers"])
    if count == 0:
        return "no_results"
    if count == 1:
        return "fetch_parse"
    return "select_paper"
