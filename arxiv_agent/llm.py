"""LLM provider wiring: OpenAI primary, Groq free-tier fallback.

Project decision: OpenAI gives the best quality for development, but the
assessment brief requires the submission to run with **no paid API key**.
This module resolves both: OpenAI is used as the primary model (retried
with exponential backoff on transient/rate-limit errors) when
`OPENAI_API_KEY` is set, falling back to Groq's free tier if it keeps
failing. When `OPENAI_API_KEY` is absent, OpenAI is skipped entirely and
Groq is used directly - so the agent runs on zero paid keys out of the box.

Model IDs are read from env vars (not hardcoded) because provider lineups
change faster than this code should need to: as of Sept 2026, OpenAI's
cost-optimized model is `gpt-5.6-luna` and Groq's recommended free-tier
replacement for the now-deprecated Llama-3.3-70B is `openai/gpt-oss-120b`
(console.groq.com/docs/deprecations). Override via OPENAI_MODEL / GROQ_MODEL
if either provider's lineup has moved on again by the time this runs.

Every call site in this project needs *structured* output (selection,
briefing, QA), so `get_llm()` takes the target Pydantic schema and binds
it via `.with_structured_output()` before wrapping with retry/fallback -
not after. `Runnable.with_retry()` returns a plain `RunnableRetry` that
does not proxy `.with_structured_output()` (verified against the
installed langchain-core: only `RunnableWithFallbacks` proxies arbitrary
attribute access), so binding the schema first and wrapping resilience
around the already-structured runnable is required, not just tidier.
"""

from __future__ import annotations

import os

from pydantic import BaseModel

from langchain_core.runnables import Runnable

from arxiv_agent.config import GROQ_API_KEY, OPENAI_API_KEY

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_MAX_PRIMARY_ATTEMPTS = 3


def _base_openai(temperature: float) -> Runnable:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model_name=DEFAULT_OPENAI_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=temperature,
    )


def _base_groq(temperature: float) -> Runnable:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model_name=DEFAULT_GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature,
    )


def _with_structured_output(runnable: Runnable, schema: type[BaseModel] | None) -> Runnable:
    return runnable.with_structured_output(schema) if schema else runnable


def _with_openai_retry(runnable: Runnable) -> Runnable:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    return runnable.with_retry(
        retry_if_exception_type=(
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
        ),
        stop_after_attempt=_MAX_PRIMARY_ATTEMPTS,
        wait_exponential_jitter=True,
    )


def get_llm(temperature: float = 0.0, *, output_schema: type[BaseModel] | None = None) -> Runnable:
    """Return a chat-model Runnable with retry + free-tier fallback baked in.

    - Both keys set: OpenAI primary (retried up to 3x with exponential
      backoff on 429/connection/timeout/5xx), falling back to Groq if the
      retries are exhausted or the response fails validation.
    - Only GROQ_API_KEY set: Groq directly - the zero-paid-key path the
      assessment brief requires.
    - Only OPENAI_API_KEY set: OpenAI with retry, no fallback available.
    - Neither set: fails fast with a clear message instead of a deep SDK
      stack trace.

    Pass `output_schema` (a Pydantic model) for any structured-output call
    - i.e. every real call site in this project - so the schema is bound
    before retry/fallback wrapping rather than after.
    """

    if not OPENAI_API_KEY and not GROQ_API_KEY:
        raise RuntimeError(
            "No LLM provider configured. Set OPENAI_API_KEY and/or GROQ_API_KEY "
            "in .env - GROQ_API_KEY alone is enough to run the agent on a free "
            "tier with no paid key."
        )

    if not OPENAI_API_KEY:
        return _with_structured_output(_base_groq(temperature), output_schema)

    primary = _with_structured_output(_base_openai(temperature), output_schema)
    primary = _with_openai_retry(primary)

    if not GROQ_API_KEY:
        return primary

    fallback = _with_structured_output(_base_groq(temperature), output_schema)
    return primary.with_fallbacks([fallback])
