"""Fetch & Parse node: download the PDF and convert it to page-level
Markdown text via pymupdf4llm.

A paper whose total extracted text falls under `_MIN_EXTRACTED_CHARS` is
treated as scanned or broken-layout - OCR is explicitly out of scope for
this assessment, so that case surfaces as a graceful failure rather than
an attempted recovery.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pymupdf4llm
import requests

from arxiv_agent.config import PDF_DIR
from arxiv_agent.schemas import ParsedPage
from arxiv_agent.state import GraphState

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_SECONDS = 60
_RETRY_DELAY_SECONDS = 2.0
_MIN_EXTRACTED_CHARS = 500


def _download_pdf(pdf_url: str, dest_path: Path) -> None:
    response = requests.get(pdf_url, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    dest_path.write_bytes(response.content)


def _download_with_retry(pdf_url: str, dest_path: Path) -> Exception | None:
    """Try twice; return None on success, else the final exception."""

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            _download_pdf(pdf_url, dest_path)
            return None
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            logger.warning("PDF download attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(_RETRY_DELAY_SECONDS)
    return last_exc


def fetch_parse(state: GraphState) -> dict:
    """Download `state["paper"].pdf_url` (one retry on failure) and parse
    it into page-level Markdown text.
    """

    paper = state["paper"]
    dest_path = PDF_DIR / f"{paper.arxiv_id}.pdf"

    download_error = _download_with_retry(paper.pdf_url, dest_path)
    if download_error is not None:
        return {
            "parse_failed": True,
            "error": (
                f"Could not download the PDF for {paper.arxiv_id} after 2 attempts: "
                f"{download_error}"
            ),
        }

    try:
        raw_pages = pymupdf4llm.to_markdown(str(dest_path), page_chunks=True)
    except Exception as exc:  # pymupdf4llm's failure modes aren't a clean typed hierarchy
        logger.warning("PDF parse failed for %s: %s", paper.arxiv_id, exc)
        return {
            "parse_failed": True,
            "error": f"Could not parse the PDF for {paper.arxiv_id}: {exc}",
        }

    pages = [
        ParsedPage(
            page_number=page["metadata"].get("page_number", i + 1),
            text=page["text"],
        )
        for i, page in enumerate(raw_pages)
    ]
    total_chars = sum(len(p.text) for p in pages)

    if total_chars < _MIN_EXTRACTED_CHARS:
        return {
            "parsed_pages": pages,
            "parse_failed": True,
            "error": (
                f"'{paper.title}' looks scanned or has a broken layout ({total_chars} "
                f"characters extracted from {len(pages)} pages). OCR is not supported "
                "- try a different paper."
            ),
        }

    return {"parsed_pages": pages, "parse_failed": False}


def route_after_fetch_parse(state: GraphState) -> str:
    """Conditional-edge router: parse failure vs. ready to chunk."""

    return "parse_failed" if state["parse_failed"] else "chunk_embed"
