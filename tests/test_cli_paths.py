"""Routing infrastructure tests call handlers directly; public runtime guards stay active."""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from jarvis import cli
from jarvis.clients import create_summary_store
from jarvis.config import load_config
from jarvis.store import DatabaseCompatibilityError, SummaryStore


@pytest.fixture
def external_source(profile_repo: Path) -> Path:
    fixture = Path(__file__).parent / "fixtures/chatgpt_tiny.json"
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    raw["conversation_id"] = "shared-conversation"
    source = profile_repo / "external/source.json"
    source.parent.mkdir()
    source.write_text(json.dumps(raw), encoding="utf-8")
    return source


def ingest_args(
    source: Path, persist: bool = False, output_dir: Path | None = None
) -> argparse.Namespace:
    return argparse.Namespace(
        source="chatgpt", file=str(source), output_dir=output_dir, persist=persist
    )


def processing_args(**overrides: Any) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            "source": "chatgpt",
            "conversation_id": "shared-conversation",
            "inbox_dir": None,
            "persist": False,
            "force": False,
            "embed": False,
            "from_segment": 0,
            "to_segment": None,
            "context_window": 2,
            "threshold": 0.5,
            "dry_run": False,
            **overrides,
        }
    )


def test_identical_ids_isolated_and_processing_does_not_need_original(
    profile_repo: Path, external_source: Path
) -> None:
    configs = [load_config(name) for name in ("personal", "mom")]
    before = external_source.read_bytes(), external_source.stat().st_mtime_ns
    checksum = hashlib.sha256(before[0]).hexdigest()
    with (
        patch("jarvis.clients.SummaryStore", side_effect=AssertionError("No persistence")),
        patch("jarvis.clients.VectorStore", side_effect=AssertionError("No vector indexing")),
        patch("jarvis.clients.EmbeddingClient", side_effect=AssertionError("No embeddings")),
    ):
        for config in configs:
            assert cli.cmd_ingest(ingest_args(external_source), config) == 0
            normalized_path = (
                Path(config["chatgpt_inbox_dir"]) / "shared-conversation/normalized.json"
            )
            normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
            assert normalized["raw_export_path"] == str(external_source)
            assert normalized["raw_export_sha256"] == checksum
            assert normalized["source_provenance"] == [
                {"original_path": str(external_source), "sha256": checksum}
            ]
            assert not Path(config["db_path"]).exists()
            assert not Path(config["output_root"]).exists()
        assert (external_source.read_bytes(), external_source.stat().st_mtime_ns) == before
        external_source.unlink()  # Synthetic source only: subsequent processing uses artifacts.
        with patch("jarvis.clients.OllamaClient") as inference:
            inference.return_value.model = "synthetic-model"
            inference.return_value.chat.return_value = ("{}", False, "")
            for config in configs:
                config["prompts_dir"] = str(Path(__file__).parents[1] / "prompts")
                inference.return_value.parse_json_response.return_value = (
                    {"summary": config["profile_name"], "confidence": 0.9},
                    False,
                    "",
                )
                assert cli.cmd_summarize_segments(processing_args(), config) == 0
        for config in configs:
            summaries = list(
                (Path(config["output_root"]) / "shared-conversation/segment_summaries").glob(
                    "*.json"
                )
            )
            assert summaries
            assert {
                json.loads(path.read_text(encoding="utf-8"))["summary"] for path in summaries
            } == {config["profile_name"]}
            assert not Path(config["db_path"]).exists()


def test_persistence_initializes_only_selected_sqlite(
    profile_repo: Path, external_source: Path
) -> None:
    config = load_config("mom")
    with patch("jarvis.clients.VectorStore", side_effect=AssertionError("No indexing")):
        assert cli.cmd_ingest(ingest_args(external_source, persist=True), config) == 0
    store = create_summary_store(config)
    assert store.get_conversation("shared-conversation") is not None
    assert store.count_all()["source_files"] == 2
    assert not Path(load_config("personal")["db_path"]).exists()
    assert not Path(config["output_root"]).exists()


@pytest.mark.parametrize("command", ["retrieve", "answer"])
@pytest.mark.parametrize("profile", ["personal", "mom"])
def test_read_commands_do_not_initialize_resources(
    profile_repo: Path, command: str, profile: str, caplog: pytest.LogCaptureFixture
) -> None:
    config = load_config(profile)
    before = sorted(profile_repo.rglob("*"))
    args = argparse.Namespace(
        query="synthetic", top_k=5, min_results=1, min_score=0.5, verbose=False, temperature=0.3
    )
    with (
        patch.object(SummaryStore, "_init_db", side_effect=AssertionError("No initialization")),
        patch("jarvis.clients.VectorStore", side_effect=AssertionError("No vector access")),
        patch("jarvis.clients.EmbeddingClient", side_effect=AssertionError("No embedding access")),
        patch("jarvis.clients.OllamaClient", side_effect=AssertionError("No inference access")),
    ):
        assert getattr(cli, f"cmd_{command}")(args, config) == 1
    assert "not initialized" in caplog.text
    assert sorted(profile_repo.rglob("*")) == before


def test_read_only_store_validates_without_writes(profile_repo: Path) -> None:
    config = load_config()
    create_summary_store(config).insert_conversation({"conversation_id": "synthetic"})
    path = Path(config["db_path"])
    before = path.read_bytes(), path.stat().st_mtime_ns
    readonly = create_summary_store(config, read_only=True)
    assert readonly.get_conversation("synthetic") is not None
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        readonly.insert_conversation({"conversation_id": "forbidden-write"})
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_missing_read_only_store_does_not_create_parent(tmp_path: Path) -> None:
    path = tmp_path / "missing/new.db"
    with pytest.raises(DatabaseCompatibilityError, match="not initialized"):
        SummaryStore(str(path), read_only=True)
    assert not path.parent.exists()


@pytest.mark.parametrize("profile", ["personal", "mom"])
def test_valid_managed_override(profile_repo: Path, external_source: Path, profile: str) -> None:
    config = load_config(profile)
    destination = config["profile_context"].inbox_root / "custom"
    assert cli.cmd_ingest(ingest_args(external_source, output_dir=destination), config) == 0
    assert (destination / "shared-conversation/normalized.json").is_file()
    assert not config["profile_context"].chatgpt_inbox_dir.exists()


def test_personal_relative_override_remains_supported(
    profile_repo: Path, external_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config()
    monkeypatch.chdir(profile_repo)
    assert (
        cli.cmd_ingest(
            ingest_args(external_source, output_dir=Path("inbox/ai_chat/chatgpt")), config
        )
        == 0
    )


@pytest.mark.parametrize("kind", ["override", "source", "identifier"])
def test_invalid_ingest_routes_fail_before_writes(
    profile_repo: Path, external_source: Path, kind: str
) -> None:
    config = load_config("mom")
    personal = load_config("personal")["profile_context"]
    args = ingest_args(external_source)
    if kind == "override":
        args.output_dir = personal.chatgpt_inbox_dir
    elif kind == "source":
        args.file = str(personal.output_root / "private.json")
    else:
        raw = json.loads(external_source.read_text(encoding="utf-8"))
        raw["conversation_id"] = "../escape"
        external_source.write_text(json.dumps(raw), encoding="utf-8")
    before = sorted(profile_repo.rglob("*"))
    assert cli.cmd_ingest(args, config) == 1
    assert sorted(profile_repo.rglob("*")) == before


@pytest.mark.parametrize(
    "stage", ["summarize_segments", "detect_topics", "extract_segments", "fragment_extracts"]
)
def test_invalid_paths_rejected_before_force_cleanup(profile_repo: Path, stage: str) -> None:
    config = load_config()
    with (
        patch("jarvis.recovery.create_summary_store", side_effect=AssertionError("No cleanup")),
        patch("jarvis.cli.create_ollama_client", side_effect=AssertionError("No inference")),
    ):
        assert (
            getattr(cli, f"cmd_{stage}")(
                processing_args(conversation_id="../mom", force=True), config
            )
            == 1
        )
    assert not Path(config["output_root"]).exists()


def test_parser_defaults_reach_personal_profile_paths(
    profile_repo: Path, external_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config()
    monkeypatch.setattr(
        sys, "argv", ["jarvis", "ingest", "chatgpt", "--file", str(external_source)]
    )
    assert cli.main() == 0
    assert (Path(config["chatgpt_inbox_dir"]) / "shared-conversation/normalized.json").exists()


@pytest.mark.parametrize(
    "stage", ["summarize_segments", "detect_topics", "extract_segments", "fragment_extracts"]
)
def test_invalid_artifacts_prevent_all_force_cleanup(profile_repo: Path, stage: str) -> None:
    config = load_config()
    directory = Path(config["output_root"]) / "shared-conversation/extracts"
    directory.mkdir(parents=True)
    artifact = directory / "extract_000.json"
    artifact.write_text('{"segment_id": "../escape"}', encoding="utf-8")
    before = artifact.read_bytes(), artifact.stat().st_mtime_ns
    with (
        patch("jarvis.recovery.create_summary_store", side_effect=AssertionError("No cleanup")),
        patch("jarvis.cli.create_ollama_client", side_effect=AssertionError("No inference")),
    ):
        assert getattr(cli, f"cmd_{stage}")(processing_args(force=True), config) == 1
    assert (artifact.read_bytes(), artifact.stat().st_mtime_ns) == before


def test_force_without_persistence_never_initializes_missing_database(profile_repo: Path) -> None:
    config = load_config()
    (Path(config["output_root"]) / "shared-conversation/extracts").mkdir(parents=True)
    before = sorted(profile_repo.rglob("*"))
    assert cli.cmd_fragment_extracts(processing_args(force=True), config) == 0
    assert sorted(profile_repo.rglob("*")) == before
    assert not Path(config["db_path"]).exists()


def test_summarize_external_source_provenance_and_no_sqlite(profile_repo: Path) -> None:
    config = load_config("mom")
    config["prompts_dir"] = str(Path(__file__).parents[1] / "prompts")
    source = profile_repo / "conversation.json"
    source.write_text(
        '[{"speaker": "user", "content": "Synthetic conversation"}]', encoding="utf-8"
    )
    before = source.read_bytes(), source.stat().st_mtime_ns
    with patch("jarvis.clients.OllamaClient") as inference:
        inference.return_value.model = "synthetic-model"
        inference.return_value.chat.return_value = ("{}", False, "")
        inference.return_value.parse_json_response.return_value = (
            {"summary": "Synthetic"},
            False,
            "",
        )
        assert (
            cli.cmd_summarize(
                argparse.Namespace(file=str(source), force=False, persist=False), config
            )
            == 0
        )
    output = Path(config["output_root"]) / "conversation/conversation.json"
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["source_provenance"] == [
        {"original_path": str(source), "sha256": hashlib.sha256(before[0]).hexdigest()}
    ]
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert not Path(config["db_path"]).exists()


def test_summary_source_cannot_be_its_own_output(profile_repo: Path) -> None:
    config = load_config()
    source = Path(config["output_root"]) / "same/same.json"
    source.parent.mkdir(parents=True)
    source.write_text("[]", encoding="utf-8")
    before = source.read_bytes(), source.stat().st_mtime_ns
    with patch("jarvis.cli.create_ollama_client", side_effect=AssertionError("No inference")):
        assert (
            cli.cmd_summarize(
                argparse.Namespace(file=str(source), force=True, persist=False), config
            )
            == 1
        )
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_ingest_source_cannot_be_a_generated_artifact(
    profile_repo: Path, external_source: Path
) -> None:
    config = load_config()
    source = Path(config["chatgpt_inbox_dir"]) / "shared-conversation/segments/segment_000.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(external_source.read_bytes())
    before = source.read_bytes(), source.stat().st_mtime_ns
    assert cli.cmd_ingest(ingest_args(source), config) == 1
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert not (source.parent.parent / "normalized.json").exists()


@pytest.mark.parametrize("command", ["retrieve", "answer"])
def test_read_commands_never_create_missing_collection(profile_repo: Path, command: str) -> None:
    config = load_config()
    create_summary_store(config)
    database = Path(config["db_path"])
    database_before = database.read_bytes(), database.stat().st_mtime_ns
    before = sorted(profile_repo.rglob("*"))
    args = argparse.Namespace(
        query="synthetic", top_k=5, min_results=1, min_score=0.5, verbose=False, temperature=0.3
    )
    with (
        patch("jarvis.vector_store.QdrantClient") as qdrant,
        patch("jarvis.clients.EmbeddingClient") as embeddings,
        patch("jarvis.clients.OllamaClient"),
    ):
        qdrant.return_value.collection_exists.return_value = False
        embeddings.return_value.embed.return_value = [1.0, 0.0]
        assert getattr(cli, f"cmd_{command}")(args, config) == 1
        qdrant.return_value.create_collection.assert_not_called()
        qdrant.return_value.upsert.assert_not_called()
    # SQLite may create reader coordination sidecars for an existing WAL database.
    created = set(profile_repo.rglob("*")) - set(before)
    assert created <= {Path(f"{database}-wal"), Path(f"{database}-shm")}
    assert (database.read_bytes(), database.stat().st_mtime_ns) == database_before
