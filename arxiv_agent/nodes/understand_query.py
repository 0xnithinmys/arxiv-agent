"""Query Understanding node: classify raw input as an arXiv ID/URL or a topic.

This is the graph's first conditional-edge decision point: arXiv ID/URL
input routes toward a direct by-ID fetch (no search ambiguity to resolve),
free-text input routes toward topic search, and empty/whitespace-only
input routes to a validation-error end state instead of reaching the
arXiv API at all.
"""

from __future__ import annotations

import re

from arxiv_agent.state import GraphState

# New-style arXiv IDs: YYMM.NNNN[N], optionally with a version suffix
# (v1, v2, ...). Matches the assessment brief's own \d{4}\.\d{4,5} spec,
# extended with the optional version suffix since that's how IDs are
# commonly copy-pasted from arxiv.org URLs. Pre-2007 old-style IDs
# (e.g. hep-th/9901001) are intentionally not matched here - see the
# README's known limitations.
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")

# arxiv.org URLs: optional scheme/www, /abs/ or /pdf/ path, optional
# .pdf suffix / trailing slash / query string - anything after the
# captured ID is ignored rather than required to match.
_ARXIV_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<arxiv_id>\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def extract_arxiv_id(raw_input: str) -> str | None:
    """Return a normalized arXiv ID if `raw_input` is a bare ID or an
    arxiv.org URL naming one, else None (meaning: treat it as a topic).
    """

    candidate = raw_input.strip()

    if _ARXIV_ID_RE.match(candidate):
        return candidate

    url_match = _ARXIV_URL_RE.match(candidate)
    if url_match:
        return url_match.group("arxiv_id")

    return None


def understand_query(state: GraphState) -> dict:
    """Classify `state["raw_input"]` and return the state fields to update.

    Empty/whitespace-only input is surfaced via `state["error"]` rather
    than raised, so the graph can route to a graceful message instead of
    crashing or wasting an arXiv API call.
    """

    stripped = state["raw_input"].strip()

    if not stripped:
        return {
            "input_type": None,
            "error": (
                "Empty input: provide a research topic, an arXiv ID "
                "(e.g. 2401.12345), or an arXiv URL "
                "(e.g. https://arxiv.org/abs/2401.12345)."
            ),
        }

    arxiv_id = extract_arxiv_id(stripped)
    if arxiv_id:
        return {"input_type": "arxiv_id", "raw_input": arxiv_id}

    return {"input_type": "topic", "raw_input": stripped}


def route_after_understanding(state: GraphState) -> str:
    """Conditional-edge router: name of the next node given the classification."""

    if state["error"]:
        return "validation_error"
    if state["input_type"] == "arxiv_id":
        return "fetch_by_id"
    return "search_papers"
