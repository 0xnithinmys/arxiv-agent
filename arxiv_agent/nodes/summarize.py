"""Executive Briefing node: one structured-output call over the full
parsed paper text.

A single full-document call (not map-reduce) is correct and token-
efficient for a single paper - the only real risk is a paper long enough
to threaten the LLM's context window, handled below with a head/tail
truncation rather than added chunking complexity.
"""

from __future__ import annotations

import logging

from arxiv_agent.llm import get_llm
from arxiv_agent.schemas import BriefingContent, ExecutiveBriefing
from arxiv_agent.state import GraphState

logger = logging.getLogger(__name__)

# Generous enough to cover the vast majority of arXiv papers (even long
# surveys) well within any current model's context window, while still
# guarding against a pathologically long "huge paper" - the brief's own
# named edge case - blowing up latency/cost.
_MAX_INPUT_CHARS = 150_000

# Used only on retry, after the full-size attempt has already failed once
# (for any reason). Groq's free tier is capped at 8,000 tokens/minute for
# openai/gpt-oss-120b - confirmed directly from a real 413 error ("Limit
# 8000, Requested 42400") triggered by an 86-page paper. Groq receives
# this exact prompt whenever it's used, whether as the automatic fallback
# after an OpenAI failure or as the sole configured provider - so a
# too-large prompt doesn't just risk a slower answer, it makes the
# fallback structurally unable to help on a large paper. ~18,000 chars
# (~5,100 tokens at the ~3.54 chars/token ratio the real error implies)
# leaves headroom under 8,000 for the system prompt and the model's own
# output tokens, which count against the same per-minute budget.
_RETRY_MAX_INPUT_CHARS = 18_000
_HEAD_FRACTION = 0.6  # intro/abstract/problem statement tend to live here

_SYSTEM_PROMPT = (
    "You are writing an executive briefing for a busy researcher deciding "
    "whether to read this paper in full. Ground every section in the "
    "provided paper text - do not invent results or claims the paper "
    "doesn't make.\n\n"
    "- why_it_matters: one paragraph, plain English.\n"
    "- problem_statement: what problem the paper is trying to solve.\n"
    "- method_approach: bullet points summarizing the technical approach.\n"
    "- key_results: bullet points on the paper's important findings.\n"
    "- limitations: bullet points on the paper's limitations. This must "
    "not be empty - if the paper has no explicit limitations section, "
    "infer reasonable ones from its scope, assumptions, or evaluation "
    "(e.g. dataset size, generalization, compute cost).\n"
    "- follow_up_questions: 3-5 questions a reader might naturally ask "
    "after reading this briefing.\n"
    "- architecture_diagram: a Mermaid flowchart (starting with "
    "'flowchart TD' or 'flowchart LR', no markdown code fences) diagramming "
    "the paper's own core method or pipeline, 8-12 nodes. Leave it an "
    "empty string if the paper has no clear pipeline/architecture to "
    "diagram - do not force one onto a pure theory or position paper."
)


def _build_full_text(state: GraphState, max_chars: int) -> str:
    full_text = "\n\n".join(page.text for page in state["parsed_pages"])
    if len(full_text) <= max_chars:
        return full_text

    head = int(max_chars * _HEAD_FRACTION)
    tail = max_chars - head
    return (
        full_text[:head]
        + "\n\n[... middle of paper omitted for length ...]\n\n"
        + full_text[-tail:]
    )


def _generate(paper, full_text: str) -> ExecutiveBriefing:
    llm = get_llm(temperature=0.0, output_schema=BriefingContent)
    content = llm.invoke(
        f"{_SYSTEM_PROMPT}\n\nPaper title: {paper.title}\n\nFull paper text:\n{full_text}"
    )
    return ExecutiveBriefing(metadata=paper, **content.model_dump())


def summarize(state: GraphState) -> dict:
    """Generate the executive briefing and store it in state.

    Tries the large, quality-preserving prompt first - the vast majority
    of papers succeed here via OpenAI. Only on failure (content-policy
    rejection, a provider's own context limit, anything) does it retry
    once with a much smaller, Groq-safe prompt, which also gives Groq's
    own moderation/limits a fair independent shot rather than always
    inheriting whatever the first attempt tripped over.
    """

    paper = state["paper"]
    last_exc: Exception | None = None

    for max_chars in (_MAX_INPUT_CHARS, _RETRY_MAX_INPUT_CHARS):
        full_text = _build_full_text(state, max_chars)
        try:
            briefing = _generate(paper, full_text)
            return {"briefing": briefing}
        except Exception as exc:
            logger.warning(
                "Briefing generation failed for %s at %d-char budget: %s",
                paper.arxiv_id, max_chars, exc,
            )
            last_exc = exc

    return {
        "error": (
            f"Could not generate a briefing for {paper.arxiv_id} even after "
            f"retrying with a shorter excerpt: {last_exc}"
        )
    }


def route_after_summarize(state: GraphState) -> str:
    """Conditional-edge router: briefing failure vs. ready for QA."""

    return "briefing_failed" if state["error"] else "await_question"
