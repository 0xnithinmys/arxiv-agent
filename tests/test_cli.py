import pytest

from arxiv_agent import cli


def test_prompt_returns_value_on_valid_input(monkeypatch):
    monkeypatch.setattr(cli.console, "input", lambda label: "2401.12345")
    assert cli._prompt("Topic: ") == "2401.12345"


def test_prompt_retries_on_empty_input(monkeypatch):
    responses = iter(["", "   ", "a real question"])
    monkeypatch.setattr(cli.console, "input", lambda label: next(responses))
    assert cli._prompt("Question: ") == "a real question"


@pytest.mark.parametrize("command", ["exit", "quit", "q", "EXIT", "Quit"])
def test_prompt_returns_none_on_exit_command(monkeypatch, command):
    monkeypatch.setattr(cli.console, "input", lambda label: command)
    assert cli._prompt("Question: ") is None


def test_prompt_returns_none_on_eof(monkeypatch):
    def _raise(label):
        raise EOFError

    monkeypatch.setattr(cli.console, "input", _raise)
    assert cli._prompt("Question: ") is None


def test_prompt_returns_none_on_keyboard_interrupt(monkeypatch):
    def _raise(label):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.console, "input", _raise)
    assert cli._prompt("Question: ") is None
