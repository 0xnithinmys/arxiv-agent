"""Pure, model-agnostic chunking: paragraph-aware packing into a token
budget with overlap, tracking the nearest markdown section header.

Kept separate from the embedding node (and free of any sentence-transformers
import) so the algorithm is unit-testable with a tiny fake tokenizer
instead of downloading the real embedding model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list: ...
    def decode(self, ids: list) -> str: ...


@dataclass
class TextChunk:
    text: str
    section: str


_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)")
_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]")


def _clean_section_title(raw: str) -> str:
    return _MARKDOWN_EMPHASIS_RE.sub("", raw).strip()


def _split_paragraphs_with_sections(text: str) -> list[tuple[str, str]]:
    """Return (paragraph_text, section_title) pairs in reading order,
    tagging each paragraph with the most recent markdown header seen.
    """

    current_section = ""
    pairs: list[tuple[str, str]] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        header_match = _HEADER_RE.match(para)
        if header_match:
            current_section = _clean_section_title(header_match.group(1))
        pairs.append((para, current_section))
    return pairs


def _hard_split_by_tokens(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Last-resort fallback for a single paragraph too long to fit in one
    chunk on its own - splits by raw token count rather than leaving a
    chunk that would silently get truncated at embedding time.
    """

    ids = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def chunk_text(
    text: str,
    tokenizer: Tokenizer,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    """Greedily pack paragraphs into ~target_tokens chunks, carrying the
    trailing ~overlap_tokens worth of paragraphs into the next chunk for
    continuity. Never lets a chunk exceed target_tokens: a single
    paragraph that alone is too long is hard-split by token count.
    """

    pairs = _split_paragraphs_with_sections(text)

    chunks: list[TextChunk] = []
    current: list[tuple[str, str, int]] = []  # (para_text, section, n_tokens)
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        chunks.append(
            TextChunk(text="\n\n".join(p[0] for p in current), section=current[0][1])
        )
        overlap: list[tuple[str, str, int]] = []
        acc = 0
        for p in reversed(current):
            if acc >= overlap_tokens:
                break
            overlap.insert(0, p)
            acc += p[2]
        current = overlap
        current_tokens = acc

    for para_text, section in pairs:
        n_tokens = len(tokenizer.encode(para_text, add_special_tokens=False))

        if n_tokens > target_tokens:
            flush()
            for piece in _hard_split_by_tokens(para_text, tokenizer, target_tokens):
                chunks.append(TextChunk(text=piece, section=section))
            continue

        if current_tokens + n_tokens > target_tokens and current:
            flush()

        current.append((para_text, section, n_tokens))
        current_tokens += n_tokens

    flush()
    return chunks
