"""Pydantic v2 schemas shared across the agent's nodes.

These are the structured-output contracts the LLM must fill and the code
must validate — they are what makes the briefing and QA grounding
verifiable rather than free-form text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata for a single arXiv paper, as required by the briefing spec."""

    title: str
    authors: list[str]
    arxiv_id: str
    published: str  # ISO date string, e.g. "2024-01-15"
    link: str
    abstract: str
    categories: list[str] = Field(default_factory=list)
    pdf_url: str | None = None


class PaperSelection(BaseModel):
    """The LLM's choice of the best-matching candidate for a topic search."""

    index: int = Field(..., description="0-based index into the candidate list.")
    justification: str = Field(..., description="One-line reason for the choice.")


class ParsedPage(BaseModel):
    """One page of a parsed PDF, kept separate (not flattened to one big
    string) so later chunking can still attribute each chunk to a page.
    """

    page_number: int  # 1-indexed, matching pymupdf4llm's own numbering
    text: str


class Chunk(BaseModel):
    """A single retrievable unit of a parsed paper."""

    chunk_id: str
    doc_id: str  # arxiv_id
    section: str = ""
    page: int | None = None
    text: str
    metadata: dict = Field(default_factory=dict)


class BriefingContent(BaseModel):
    """The LLM-generated portion of the executive briefing.

    Metadata is deliberately excluded: title/authors/date/link are already
    known exactly from the arXiv API (PaperMetadata), so they are injected
    by the node rather than asked of the model - which would risk the LLM
    subtly re-deriving (and getting slightly wrong) data that's already
    ground truth. `limitations` has `min_length=1` so the model cannot
    silently skip the section - the brief is explicit that limitations
    must never be omitted.
    """

    why_it_matters: str = Field(..., description="One-paragraph plain-English summary.")
    problem_statement: str
    method_approach: list[str] = Field(..., min_length=1, description="Bullet points.")
    key_results: list[str] = Field(..., min_length=1)
    limitations: list[str] = Field(..., min_length=1)
    follow_up_questions: list[str] = Field(default_factory=list)
    architecture_diagram: str = Field(
        default="",
        description=(
            "A Mermaid flowchart diagramming the paper's own core method or "
            "pipeline (not this agent's architecture) - 8-12 nodes, valid "
            "Mermaid flowchart syntax only, no markdown code fences, "
            "starting with 'flowchart TD' or 'flowchart LR'. Empty string "
            "if the paper has no clear pipeline/architecture to diagram "
            "(e.g. a pure theory or position paper) - never force one."
        ),
    )


class ExecutiveBriefing(BaseModel):
    """The structured executive briefing the assessment brief requires."""

    metadata: PaperMetadata
    why_it_matters: str
    problem_statement: str
    method_approach: list[str]
    key_results: list[str]
    limitations: list[str]
    follow_up_questions: list[str] = Field(default_factory=list)
    architecture_diagram: str = ""

    def to_markdown(self) -> str:
        """Render per the brief's exact required section headings, plus
        the (optional) architecture diagram as a fenced ```mermaid block -
        standard GitHub/many-renderers markdown, so it's a real diagram
        wherever this is viewed, not just in the web UI.
        """

        lines = [
            f"# {self.metadata.title}",
            "",
            "## Metadata",
            f"- **Title:** {self.metadata.title}",
            f"- **Authors:** {', '.join(self.metadata.authors)}",
            f"- **arXiv ID:** {self.metadata.arxiv_id}",
            f"- **Publish date:** {self.metadata.published}",
            f"- **Link:** {self.metadata.link}",
            "",
            "## Why This Paper Matters",
            self.why_it_matters,
            "",
            "## Problem Statement",
            self.problem_statement,
            "",
            "## Method / Approach",
            *[f"- {item}" for item in self.method_approach],
        ]
        if self.architecture_diagram:
            lines += [
                "",
                "## Architecture Diagram",
                "```mermaid",
                self.architecture_diagram,
                "```",
            ]
        lines += [
            "",
            "## Key Results / Claims",
            *[f"- {item}" for item in self.key_results],
            "",
            "## Limitations",
            *[f"- {item}" for item in self.limitations],
            "",
            "## Suggested Follow-Up Questions",
            *[f"- {item}" for item in self.follow_up_questions],
        ]
        return "\n".join(lines)


class QAAnswer(BaseModel):
    """A grounded answer to a follow-up question about the paper.

    `citations` lists the chunk_ids the model claims support the answer;
    the QA node must verify each one was actually in the retrieved set
    before trusting `is_grounded`.
    """

    answer: str
    citations: list[str] = Field(default_factory=list)
    is_grounded: bool
