import pytest

from arxiv_agent.nodes.understand_query import (
    extract_arxiv_id,
    route_after_understanding,
    understand_query,
)
from arxiv_agent.state import create_initial_state


@pytest.mark.parametrize(
    "raw_input,expected_id",
    [
        ("2401.12345", "2401.12345"),
        ("2401.12345v2", "2401.12345v2"),
        ("0706.1234", "0706.1234"),
        ("  2401.12345  ", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("http://arxiv.org/abs/2401.12345", "2401.12345"),
        ("arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://www.arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345.pdf", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345v2/", "2401.12345v2"),
    ],
)
def test_extract_arxiv_id_matches(raw_input, expected_id):
    assert extract_arxiv_id(raw_input) == expected_id


@pytest.mark.parametrize(
    "raw_input",
    [
        "recent work on KV-cache compression for LLMs",
        "check out 2401.12345 please",
        "2401.999999",
        "arxiv.org",
        "https://arxiv.org/abs/",
        "",
        "   ",
    ],
)
def test_extract_arxiv_id_no_match(raw_input):
    assert extract_arxiv_id(raw_input) is None


def test_understand_query_classifies_arxiv_id():
    state = create_initial_state("2401.12345")
    update = understand_query(state)
    assert update["input_type"] == "arxiv_id"
    assert update["raw_input"] == "2401.12345"
    assert "error" not in update


def test_understand_query_classifies_url_and_normalizes_to_bare_id():
    state = create_initial_state("https://arxiv.org/abs/2401.12345v1")
    update = understand_query(state)
    assert update["input_type"] == "arxiv_id"
    assert update["raw_input"] == "2401.12345v1"


def test_understand_query_classifies_topic():
    topic = "recent work on KV-cache compression for LLMs"
    state = create_initial_state(topic)
    update = understand_query(state)
    assert update["input_type"] == "topic"
    assert update["raw_input"] == topic


def test_understand_query_flags_empty_input():
    state = create_initial_state("   ")
    update = understand_query(state)
    assert update["input_type"] is None
    assert "Empty input" in update["error"]


def test_route_after_understanding():
    assert route_after_understanding({"error": "x", "input_type": None}) == "validation_error"
    assert route_after_understanding({"error": None, "input_type": "arxiv_id"}) == "fetch_by_id"
    assert route_after_understanding({"error": None, "input_type": "topic"}) == "search_papers"
