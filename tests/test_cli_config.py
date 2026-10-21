"""CLI inspection and temporary runtime safety boundary regression tests."""

import sqlite3
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis import cli
from jarvis.config import load_config
from jarvis.profiles import ConfigurationError
from jarvis.web.app import create_app


def test_config_show_has_no_storage_or_service_side_effects(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["jarvis", "--profile", "mom", "config", "show"])
    with ExitStack() as stack:
        for target in (
            "jarvis.cli.SummaryStore",
            "jarvis.cli.VectorStore",
            "jarvis.cli.OllamaClient",
            "jarvis.cli.EmbeddingClient",
            "jarvis.cli._build_memory_layer",
            "jarvis.web.app.create_app",
            "requests.sessions.Session.request",
        ):
            stack.enter_context(patch(target, side_effect=AssertionError(target)))
        stack.enter_context(patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")))
        stack.enter_context(patch.object(sqlite3, "connect", side_effect=AssertionError("SQLite")))
        assert cli.main() == 0
    output = capsys.readouterr().out
    assert "Profile: mom" in output
    assert "jarvis_mom_fragments" in output
    assert str(profile_repo / "data/profiles/mom/jarvis.db") in output
    assert not (profile_repo / "data").exists()


@pytest.mark.parametrize(
    ("arguments", "handler"),
    [
        (["serve"], "cmd_serve"),
        (["retrieve", "--query", "test"], "cmd_retrieve"),
        (["answer", "test"], "cmd_answer"),
        (["summarize", "--file", "external.json"], "cmd_summarize"),
        (["ingest", "chatgpt", "--file", "external.json"], "cmd_ingest"),
        (["summarize-segments", "chatgpt", "--conversation-id", "test"], "cmd_summarize_segments"),
        (["detect-topics", "chatgpt", "--conversation-id", "test"], "cmd_detect_topics"),
        (["extract-segments", "chatgpt", "--conversation-id", "test"], "cmd_extract_segments"),
        (["fragment-extracts", "chatgpt", "--conversation-id", "test"], "cmd_fragment_extracts"),
    ],
)
def test_non_personal_commands_fail_before_dispatch(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str], handler: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["jarvis", "--profile", "mom", *arguments])
    with patch.object(cli, handler) as command, pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    command.assert_not_called()
    assert not (profile_repo / "data").exists()


def test_environment_selected_profile_is_also_guarded(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JARVIS_PROFILE", "mom")
    monkeypatch.setattr(sys, "argv", ["jarvis", "serve"])
    with patch.object(cli, "cmd_serve") as serve, pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    serve.assert_not_called()


def test_invalid_config_is_cli_error_two(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["jarvis", "--profile", "unknown", "config", "show"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Unknown profile" in capsys.readouterr().err


@pytest.mark.parametrize("explicit", [False, True])
def test_personal_command_still_dispatches(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    prefix = ["--profile", "personal"] if explicit else []
    monkeypatch.setattr(sys, "argv", ["jarvis", *prefix, "serve"])
    with patch.object(cli, "cmd_serve", return_value=0) as serve:
        assert cli.main() == 0
    assert serve.call_args.args[1]["profile_name"] == "personal"


@pytest.mark.parametrize("preloaded", [False, True])
def test_web_factory_blocks_non_personal_before_creating_directories(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, preloaded: bool
) -> None:
    monkeypatch.setenv("JARVIS_PROFILE", "mom")
    config = load_config("mom") if preloaded else None
    with (
        patch.object(Path, "mkdir", side_effect=AssertionError("Unexpected mkdir")),
        patch("jarvis.web.app.Flask", side_effect=AssertionError("Unexpected app creation")),
        pytest.raises(ConfigurationError, match="config show"),
    ):
        create_app(config)
    assert not (profile_repo / "data").exists()
