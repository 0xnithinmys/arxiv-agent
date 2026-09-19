from arxiv_agent.nodes.qa import (
    _metadata_chunk,
    answer_question,
    prepare_regeneration,
    record_turn,
    refuse,
    retrieve_chunks,
    route_after_answer,
    route_after_retrieve,
    route_after_verify,
    verify_answer,
)
from arxiv_agent.schemas import Chunk, PaperMetadata, QAAnswer
from arxiv_agent.state import create_initial_state


def _make_paper() -> PaperMetadata:
    return PaperMetadata(
        title="Test Paper",
        authors=["A. Author"],
        arxiv_id="2401.12345",
        published="2024-01-15",
        link="https://arxiv.org/abs/2401.12345",
        abstract="abstract",
        categories=["cs.CL"],
        pdf_url="https://arxiv.org/pdf/2401.12345",
    )


def _make_chunks(ids: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=cid, doc_id="2401.12345", section="Intro", page=1, text=f"text for {cid}")
        for cid in ids
    ]


def _state_with_question(question="what does it do?"):
    state = create_initial_state("2401.12345")
    state["paper"] = _make_paper()
    state["question"] = question
    return state


# --- retrieve_chunks -------------------------------------------------------


def test_retrieve_chunks_success(monkeypatch):
    monkeypatch.setattr("arxiv_agent.nodes.qa.embed_texts", lambda texts: [[0.1, 0.2]])

    class FakeCollection:
        def query(self, query_embeddings, n_results, include):
            return {
                "ids": [["c1", "c2"]],
                "documents": [["doc one", "doc two"]],
                "metadatas": [[
                    {"doc_id": "2401.12345", "section": "Intro", "page": 1},
                    {"doc_id": "2401.12345", "section": "Method", "page": 3},
                ]],
            }

    monkeypatch.setattr(
        "arxiv_agent.nodes.qa.get_existing_collection", lambda arxiv_id: FakeCollection()
    )

    update = retrieve_chunks(_state_with_question())
    retrieved = update["retrieved_chunks"]
    # the 2 real matches, plus the always-included synthetic metadata chunk
    assert len(retrieved) == 3
    assert retrieved[0].chunk_id == "c1"
    assert retrieved[1].section == "Method"
    assert retrieved[2].chunk_id == "2401.12345#metadata"
    assert "A. Author" in retrieved[2].text
    assert "error" not in update


def test_retrieve_chunks_failure(monkeypatch):
    def _raise(texts):
        raise RuntimeError("model not loaded")

    monkeypatch.setattr("arxiv_agent.nodes.qa.embed_texts", _raise)
    update = retrieve_chunks(_state_with_question())
    assert "error" in update


def test_route_after_retrieve():
    assert route_after_retrieve({"error": "boom"}) == "retrieval_failed"
    assert route_after_retrieve({"error": None}) == "answer_question"


# --- metadata chunk: fixes authors/title/date questions getting answered ---
# --- from a retrieved *references-section* chunk about a DIFFERENT paper ---


def test_metadata_chunk_contains_paper_identity():
    chunk = _metadata_chunk(_make_paper())
    assert chunk.chunk_id == "2401.12345#metadata"
    assert chunk.section == "Metadata"
    assert "A. Author" in chunk.text
    assert "Test Paper" in chunk.text
    assert "2024-01-15" in chunk.text


def test_answer_question_prompt_anchors_paper_identity(monkeypatch):
    captured = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return QAAnswer(answer="x", citations=["c1"], is_grounded=True)

    monkeypatch.setattr("arxiv_agent.nodes.qa.get_llm", lambda **kwargs: FakeLLM())

    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1"])
    answer_question(state)

    assert "Test Paper" in captured["prompt"]
    assert "2401.12345" in captured["prompt"]
    assert "never a different paper" in captured["prompt"]


def test_metadata_chunk_resolves_the_wrong_paper_authors_failure_mode():
    """Regression test for a real bug: asking "who are the authors" on a
    paper whose references section lists many other papers' authors
    could retrieve those bibliography chunks instead of this paper's own
    title-page chunk, and the LLM would (correctly, given only bad
    context) report the wrong authors while still passing verification,
    since the citation *was* genuinely in the retrieved set. The metadata
    chunk gives the model a correct, always-retrievable source to cite
    instead.
    """

    paper = _make_paper()
    # simulates retrieval pulling bibliography chunks about OTHER papers,
    # exactly like the real failure - plus the metadata chunk this fix adds
    bibliography_chunk = Chunk(
        chunk_id="2401.12345#p12_3",
        doc_id="2401.12345",
        section="References",
        page=12,
        text="Ashish Vaswani, Noam Shazeer, ... Attention is all you need. NeurIPS, 2017.",
    )
    retrieved = [bibliography_chunk, _metadata_chunk(paper)]

    correct_answer = QAAnswer(
        answer=f"The authors are {', '.join(paper.authors)}.",
        citations=["2401.12345#metadata"],
        is_grounded=True,
    )

    state = _state_with_question("who are the authors of the paper?")
    state["retrieved_chunks"] = retrieved
    state["answer"] = correct_answer

    update = verify_answer(state)
    assert update["answer"].is_grounded is True


# --- answer_question ---------------------------------------------------------


def test_answer_question_success(monkeypatch):
    expected = QAAnswer(answer="It does X.", citations=["c1"], is_grounded=True)

    class FakeLLM:
        def invoke(self, prompt):
            return expected

    monkeypatch.setattr("arxiv_agent.nodes.qa.get_llm", lambda **kwargs: FakeLLM())

    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1", "c2"])
    update = answer_question(state)
    assert update["answer"] == expected


def test_answer_question_failure(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt):
            raise RuntimeError("provider down")

    monkeypatch.setattr("arxiv_agent.nodes.qa.get_llm", lambda **kwargs: FakeLLM())

    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1"])
    update = answer_question(state)
    assert "error" in update
    assert "answer" not in update


def test_route_after_answer():
    assert route_after_answer({"error": "boom"}) == "answer_failed"
    assert route_after_answer({"error": None}) == "verify_answer"


# --- verify_answer (the core correctness-critical logic) -------------------


def test_verify_answer_passes_when_citations_are_valid_and_grounded():
    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1", "c2", "c3"])
    state["answer"] = QAAnswer(answer="It does X.", citations=["c1", "c2"], is_grounded=True)

    update = verify_answer(state)
    assert update["answer"].is_grounded is True


def test_verify_answer_fails_on_empty_citations_even_if_llm_claims_grounded():
    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1", "c2"])
    state["answer"] = QAAnswer(answer="It does X.", citations=[], is_grounded=True)

    update = verify_answer(state)
    assert update["answer"].is_grounded is False


def test_verify_answer_fails_on_hallucinated_citation_not_in_retrieved_set():
    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1", "c2"])
    state["answer"] = QAAnswer(
        answer="It does X.", citations=["c1", "c99-does-not-exist"], is_grounded=True
    )

    update = verify_answer(state)
    assert update["answer"].is_grounded is False


def test_verify_answer_respects_llm_self_reported_ungrounded():
    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1", "c2"])
    # citations are technically valid, but the model itself said it couldn't answer
    state["answer"] = QAAnswer(
        answer="I couldn't find this in the paper.", citations=["c1"], is_grounded=False
    )

    update = verify_answer(state)
    assert update["answer"].is_grounded is False


def test_verify_answer_rejects_metadata_chunk_as_grounding_for_unrelated_claim():
    """Regression test for a real failure observed live: a question about
    something the paper never discusses (Bitcoin's price) got an answer
    that self-reported is_grounded=True citing only the always-present
    metadata chunk - membership alone made that citation "valid" even
    though the metadata chunk (title/authors/date/id/categories) has
    nothing to do with the claim. The fix requires the answer text to
    actually contain one of the metadata chunk's own fields when it's the
    only thing cited.
    """

    paper = _make_paper()
    state = _state_with_question("What was Bitcoin's price when this was published?")
    state["retrieved_chunks"] = [_metadata_chunk(paper)]
    state["answer"] = QAAnswer(
        answer="The excerpts do not mention Bitcoin's price.",
        citations=["2401.12345#metadata"],
        is_grounded=True,
    )

    update = verify_answer(state)
    assert update["answer"].is_grounded is False


def test_verify_answer_accepts_metadata_chunk_when_answer_uses_its_fields():
    paper = _make_paper()
    state = _state_with_question("what is the arxiv id?")
    state["retrieved_chunks"] = [_metadata_chunk(paper)]
    state["answer"] = QAAnswer(
        answer="The arXiv ID is 2401.12345.",
        citations=["2401.12345#metadata"],
        is_grounded=True,
    )

    update = verify_answer(state)
    assert update["answer"].is_grounded is True


def test_verify_answer_preserves_answer_text_and_citations():
    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1"])
    state["answer"] = QAAnswer(answer="It does X.", citations=["c1"], is_grounded=True)

    update = verify_answer(state)
    assert update["answer"].answer == "It does X."
    assert update["answer"].citations == ["c1"]


# --- bounded regeneration & refusal (Task 10) -------------------------------


def test_answer_question_uses_plain_prompt_on_first_attempt(monkeypatch):
    captured = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return QAAnswer(answer="x", citations=["c1"], is_grounded=True)

    monkeypatch.setattr("arxiv_agent.nodes.qa.get_llm", lambda **kwargs: FakeLLM())

    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1"])
    assert state["regeneration_count"] == 0

    answer_question(state)
    assert "IMPORTANT" not in captured["prompt"]


def test_answer_question_uses_strict_prompt_on_retry(monkeypatch):
    captured = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return QAAnswer(answer="x", citations=["c1"], is_grounded=True)

    monkeypatch.setattr("arxiv_agent.nodes.qa.get_llm", lambda **kwargs: FakeLLM())

    state = _state_with_question()
    state["retrieved_chunks"] = _make_chunks(["c1"])
    state["regeneration_count"] = 1

    answer_question(state)
    assert "IMPORTANT" in captured["prompt"]
    assert "previous attempt" in captured["prompt"]


def test_route_after_verify_returns_answer_when_grounded():
    state = _state_with_question()
    state["answer"] = QAAnswer(answer="x", citations=["c1"], is_grounded=True)
    state["regeneration_count"] = 0
    assert route_after_verify(state) == "return_answer"


def test_route_after_verify_regenerates_when_ungrounded_and_retry_available():
    state = _state_with_question()
    state["answer"] = QAAnswer(answer="x", citations=[], is_grounded=False)
    state["regeneration_count"] = 0
    assert route_after_verify(state) == "regenerate"


def test_route_after_verify_refuses_when_retry_exhausted():
    state = _state_with_question()
    state["answer"] = QAAnswer(answer="x", citations=[], is_grounded=False)
    state["regeneration_count"] = 1
    assert route_after_verify(state) == "refuse"


def test_prepare_regeneration_increments_counter():
    state = _state_with_question()
    state["regeneration_count"] = 0
    update = prepare_regeneration(state)
    assert update["regeneration_count"] == 1


def test_refuse_returns_expected_message_and_no_citations():
    update = refuse(_state_with_question())
    assert update["answer"].answer == "I couldn't find support for that in the paper."
    assert update["answer"].citations == []
    assert update["answer"].is_grounded is False


def test_record_turn_appends_without_mutating_existing_history():
    state = _state_with_question("Q1?")
    state["answer"] = QAAnswer(answer="A1", citations=["c1"], is_grounded=True)
    existing_turn = {"question": "Q0?", "answer": QAAnswer(answer="A0", citations=[], is_grounded=False)}
    state["conversation_history"] = [existing_turn]

    update = record_turn(state)
    assert len(update["conversation_history"]) == 2
    assert update["conversation_history"][0] == existing_turn
    assert update["conversation_history"][1]["question"] == "Q1?"
    # original list object must not have been mutated in place
    assert state["conversation_history"] == [existing_turn]
