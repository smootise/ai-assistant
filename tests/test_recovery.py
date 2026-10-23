"""Scoped cleanup and interrupted-run recovery against synthetic profile stores."""

import argparse
import gc
import json
import sys
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from jarvis import cli
from jarvis.artifacts import atomic_json, fragment_outputs
from jarvis.clients import build_memory_layer, create_summary_store
from jarvis.config import load_config
from jarvis.memory import MemoryLayer
from jarvis.fragmenter import Fragmenter, _MAX_STATEMENTS_PER_BATCH
from jarvis.recovery import CleanupService, check_search_ready
from jarvis.store import SummaryStore
from jarvis.vector_store import fragment_point_id


def arguments(config: dict, **overrides: Any) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            "source": "chatgpt",
            "conversation_id": "shared",
            "inbox_dir": config["chatgpt_inbox_dir"],
            "from_segment": None,
            "to_segment": None,
            "force": False,
            "persist": False,
            "embed": False,
            "dry_run": False,
            "threshold": 0.65,
            "context_window": 3,
            "retries": 1,
            **overrides,
        }
    )


def seed(config: dict, conversation: str = "shared", count: int = 1) -> SummaryStore:
    config["prompts_dir"] = str(Path(__file__).parents[1] / "prompts")
    store = create_summary_store(config)
    memory = MemoryLayer(store)
    store.insert_conversation({"conversation_id": conversation, "source_platform": "chatgpt"})
    for index in range(count):
        segment_id = f"{conversation}_s{index:03d}"
        segment = {
            "segment_id": segment_id,
            "segment_index": index,
            "conversation_id": conversation,
            "start_position": 0,
            "end_position": 1,
            "message_ids": [],
            "segment_text": "user: hello\n\nassistant: hi",
        }
        source = (
            Path(config["chatgpt_inbox_dir"])
            / conversation
            / "segments"
            / f"segment_{index:03d}.json"
        )
        atomic_json(source, segment)
        store.insert_segment(segment)
        common = {
            "segment_id": segment_id,
            "segment_index": index,
            "parent_conversation_id": conversation,
            "status": "ok",
            "source_file": f"{segment_id}.json",
            "created_at": "2026-01-01T00:00:00Z",
            "model": "synthetic",
            "latency_ms": 0,
        }
        summary = {
            **common,
            "source_kind": "ai_chat_segment",
            "summary": "old summary",
            "bullets": [],
            "action_items": [],
            "confidence": 0.8,
        }
        extract = {
            **common,
            "source_kind": "ai_chat_extract",
            "statements": [
                {"statement_index": 0, "speaker": "user", "text": "hello"},
                {"statement_index": 1, "speaker": "assistant", "text": "hi"},
            ],
        }
        fragment = {
            **common,
            "source_kind": "ai_chat_fragment",
            "fragment_index": 0,
            "text": "user: hello\nassistant: hi",
            "title": "old fragment",
            "statements": extract["statements"],
        }
        root = Path(config["output_root"]) / conversation
        atomic_json(root / "segment_summaries" / f"{segment_id}.json", summary)
        atomic_json(root / "extracts" / f"extract_{index:03d}.json", extract)
        atomic_json(root / "fragments" / f"segment_{index:03d}" / "fragment_000.json", fragment)
        for path in root.rglob("*.json"):
            if not path.with_suffix(".md").exists():
                path.with_suffix(".md").write_text("old markdown", encoding="utf-8")
        memory.persist_segment_summary(summary)
        extract_id = memory.persist_extract_with_statements(extract)
        fragment_id = memory.persist_fragment_with_links(fragment, extract_id)
        store.update_fragment_embedding(fragment_id, str(UUID(int=index + 1)), "synthetic")
    memory.persist_topic_summary(
        {"parent_conversation_id": conversation, "topic_index": 0, "summary": "old topic"},
        [f"{conversation}_s000"],
    )
    atomic_json(
        Path(config["output_root"]) / conversation / "topic_summaries" / "topic_000.json",
        {
            "source_kind": "ai_chat_topic",
            "parent_conversation_id": conversation,
            "topic_index": 0,
            "summary": "old topic",
            "status": "ok",
        },
    )
    return store


def model(stage: str) -> MagicMock:
    client = MagicMock()
    client.model = "synthetic"
    statements = [{"speaker": "user", "text": "hello"}, {"speaker": "assistant", "text": "hi"}]
    response = {
        "summaries": {
            "summary": "new summary",
            "bullets": [],
            "action_items": [],
            "confidence": 0.8,
        },
        "extracts": {"statements": statements},
        "fragments": {"fragments": [{"title": "new fragment", "statements": statements}]},
    }
    raw = json.dumps(response[stage])
    client.chat.return_value = (raw, False, "")
    client.generate_json.return_value = (raw, False, "")
    client.parse_json_response.side_effect = lambda text: (json.loads(text), False, "")
    return client


def snapshot_files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "stage,command",
    [
        ("summaries", "summarize_segments"),
        ("extracts", "extract_segments"),
        ("fragments", "fragment_extracts"),
    ],
)
@pytest.mark.parametrize("profile", ["personal", "mom"])
def test_force_range_and_dependency_isolation(
    profile_repo: Path, stage: str, command: str, profile: str
) -> None:
    config = load_config(profile)
    other = load_config("mom" if profile == "personal" else "personal")
    store = seed(config, count=31)
    seed(config, "unrelated")
    other_store = seed(other)
    other_rows = other_store.get_fragments_for_conversation("shared")
    other_before = snapshot_files(Path(other["output_root"]))
    unrelated = Path(config["output_root"]) / "unrelated"
    unrelated_before = snapshot_files(unrelated)
    root = Path(config["output_root"]) / "shared"
    before = snapshot_files(root)
    vectors = MagicMock()
    with (
        patch("jarvis.recovery.create_vector_store", return_value=vectors) as factory,
        patch("jarvis.cli.create_ollama_client", return_value=model(stage)),
    ):
        assert (
            getattr(cli, f"cmd_{command}")(
                arguments(config, force=True, from_segment=10, to_segment=20), config
            )
            == 0
        )
    if stage == "summaries":
        factory.assert_not_called()
        assert len(store.get_fragments_for_conversation("shared")) == 31
    else:
        vectors.delete_points.assert_called_once_with([str(UUID(int=i + 1)) for i in range(10, 21)])
        assert len(store.get_fragments_for_conversation("shared")) == 20
        for index in range(10, 21):
            row = store.get_extract_by_segment(f"shared_s{index:03d}")
            assert (row is None) == (stage == "extracts")
    after = snapshot_files(root)
    for index in [*range(10), *range(21, 31)]:
        for name, content in before.items():
            if any(
                part.startswith(
                    (f"shared_s{index:03d}", f"extract_{index:03d}", f"segment_{index:03d}")
                )
                for part in Path(name).parts
            ):
                assert after[name] == content
    if stage == "summaries":
        assert not (root / "topic_summaries/topic_000.json").exists()
        assert (
            sum(store.get_segment_summary(f"shared_s{i:03d}") is not None for i in range(31)) == 20
        )
    assert snapshot_files(unrelated) == unrelated_before
    assert snapshot_files(Path(other["output_root"])) == other_before
    assert other_store.get_fragments_for_conversation("shared") == other_rows


@pytest.mark.parametrize("failure", ["sqlite", "qdrant", "filesystem"])
def test_pending_cleanup_replays_without_force(profile_repo: Path, failure: str) -> None:
    config = load_config("mom")
    store = seed(config)
    vectors = MagicMock()
    client = model("extracts")
    original_invalidate = SummaryStore.invalidate_stage
    original_unlink = Path.unlink

    def interrupted_sqlite(self: SummaryStore, *args: Any) -> None:
        original_invalidate(self, *args)
        raise RuntimeError("interrupted after SQLite cleanup")

    def interrupted_filesystem(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name.startswith("cleanup_"):
            raise OSError("interrupted after filesystem cleanup")
        original_unlink(path, *args, **kwargs)

    with patch("jarvis.recovery.create_vector_store", return_value=vectors):
        if failure == "sqlite":
            with patch.object(SummaryStore, "invalidate_stage", interrupted_sqlite):
                assert cli.cmd_extract_segments(arguments(config, force=True), config) == 1
        elif failure == "qdrant":
            vectors.delete_points.side_effect = RuntimeError("Qdrant delete failed")
            assert cli.cmd_extract_segments(arguments(config, force=True), config) == 1
            vectors.delete_points.side_effect = None
        else:
            with patch.object(Path, "unlink", interrupted_filesystem):
                assert cli.cmd_extract_segments(arguments(config, force=True), config) == 1
        assert store.get_extract_by_segment("shared_s000") is None
        journal = next(CleanupService(config).directory.glob("cleanup_*.json"))
        assert json.loads(journal.read_text())["point_ids"] == [str(UUID(int=1))]
        with pytest.raises(RuntimeError, match="Pending cleanup"):
            check_search_ready(config)
        with patch("jarvis.cli.create_ollama_client", return_value=client):
            assert cli.cmd_extract_segments(arguments(config, persist=True), config) == 0
            assert cli.cmd_extract_segments(arguments(config, persist=True), config) == 0
        assert not journal.exists()
        assert client.generate_json.call_count == 1
        assert len(store.get_statements_for_extract("shared_s000_x")) == 2
        assert not store.get_fragments_for_conversation("shared")


@pytest.mark.parametrize("start,end", [(-1, None), (0, -1), (20, 10)])
@pytest.mark.parametrize("command", ["summarize_segments", "extract_segments", "fragment_extracts"])
def test_invalid_ranges_do_not_mutate(
    profile_repo: Path, start: int, end: int | None, command: str
) -> None:
    config = load_config("mom")
    seed(config)
    before = snapshot_files(Path(config["output_root"]))
    with patch("jarvis.recovery.create_vector_store") as vectors:
        assert (
            getattr(cli, f"cmd_{command}")(
                arguments(config, force=True, from_segment=start, to_segment=end), config
            )
            == 1
        )
    vectors.assert_not_called()
    assert snapshot_files(Path(config["output_root"])) == before


def test_valid_outputs_repair_missing_rows_statements_links_and_markdown(
    profile_repo: Path,
) -> None:
    config = load_config("mom")
    store = seed(config)
    with store._connect() as connection:
        connection.execute("DELETE FROM fragment_statement_links")
        connection.execute("DELETE FROM extract_statements WHERE statement_index = 1")
        connection.execute("DELETE FROM segment_summaries")
    for path in Path(config["output_root"]).rglob("*.md"):
        path.unlink()
    clients = []

    def inference(cfg: dict) -> MagicMock:
        client = MagicMock()
        clients.append(client)
        return client

    with (
        patch("jarvis.cli.create_ollama_client", side_effect=inference),
        patch("jarvis.recovery.create_vector_store", side_effect=AssertionError("No vectors")),
    ):
        for _ in range(2):
            assert cli.cmd_summarize_segments(arguments(config, persist=True), config) == 0
            assert cli.cmd_extract_segments(arguments(config, persist=True), config) == 0
            assert cli.cmd_fragment_extracts(arguments(config, persist=True), config) == 0
        for client in clients:
            client.chat.assert_not_called()
            client.generate_json.assert_not_called()
    assert store.get_segment_summary("shared_s000") is not None
    assert len(store.get_statements_for_extract("shared_s000_x")) == 2
    assert store.get_link_count("shared_s000_x_f000") == 2
    root = Path(config["output_root"]) / "shared"
    for path in root.rglob("*.json"):
        if path.name != "topic_000.json":
            assert path.with_suffix(".md").exists()


def test_upsert_success_before_metadata_failure_is_recoverable(profile_repo: Path) -> None:
    config = load_config("mom")
    store = seed(config)
    fragment_id = "shared_s000_x_f000"
    with store._connect() as connection:
        connection.execute("UPDATE fragments SET qdrant_point_id = NULL, embedded_at = NULL")
    points = {}
    vectors = MagicMock()

    def upsert(**kwargs: Any) -> str:
        points[kwargs["point_id"]] = kwargs["payload"]
        return kwargs["point_id"]

    vectors.upsert.side_effect = upsert
    embedder = MagicMock(model="synthetic")
    embedder.embed.return_value = [1.0, 0.0]
    memory = MemoryLayer(store, vectors, embedder)
    path = Path(config["output_root"]) / "shared/fragments/segment_000/fragment_000.json"
    document = json.loads(path.read_text())
    with patch.object(store, "update_fragment_embedding", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            memory.index_fragment_in_qdrant(fragment_id, document)
    assert store.get_fragment(fragment_id)["qdrant_point_id"] == fragment_point_id(fragment_id)
    assert store.get_fragment(fragment_id)["embedded_at"] is None
    with (
        patch("jarvis.cli._build_memory_layer", return_value=memory),
        patch("jarvis.cli.create_ollama_client", return_value=MagicMock()),
    ):
        assert cli.cmd_fragment_extracts(arguments(config, persist=True, embed=True), config) == 0
        assert cli.cmd_fragment_extracts(arguments(config, persist=True, embed=True), config) == 0
    assert len(points) == 1
    assert vectors.upsert.call_count == 2
    assert store.get_fragment(fragment_id)["embedded_at"] is not None


def test_dry_run_force_is_non_destructive(profile_repo: Path) -> None:
    config = load_config("mom")
    seed(config)
    before = snapshot_files(Path(config["output_root"]))
    database = Path(config["db_path"]).read_bytes()
    embedder = MagicMock()
    embedder.embed.return_value = [1.0, 0.0]
    with (
        patch("jarvis.cli.create_embedding_client", return_value=embedder),
        patch("jarvis.cli.create_ollama_client", return_value=MagicMock()),
        patch("jarvis.recovery.create_vector_store", side_effect=AssertionError("No cleanup")),
    ):
        assert cli.cmd_detect_topics(arguments(config, dry_run=True, force=True), config) == 0
    assert snapshot_files(Path(config["output_root"])) == before
    assert Path(config["db_path"]).read_bytes() == database


def test_fragment_batch_survives_interrupted_artifact_writes(profile_repo: Path) -> None:
    config = load_config("mom")
    seed(config)
    client = model("fragments")
    with (
        patch("jarvis.cli.create_ollama_client", return_value=client),
        patch("jarvis.recovery.create_vector_store", return_value=MagicMock()),
    ):
        with patch("jarvis.fragmenter.atomic_text", side_effect=OSError("interrupted Markdown")):
            assert cli.cmd_fragment_extracts(arguments(config, force=True), config) == 1
        assert cli.cmd_fragment_extracts(arguments(config), config) == 0
    assert client.chat.call_count == 1
    extract = json.loads(
        (Path(config["output_root"]) / "shared/extracts/extract_000.json").read_text()
    )
    directory = Path(config["output_root"]) / "shared/fragments/segment_000"
    assert fragment_outputs(directory, extract) is not None
    assert (directory / "fragment_000.md").exists()


@pytest.mark.parametrize("state", ["missing", "truncated", "partial"])
@pytest.mark.parametrize(
    "stage,command,relative",
    [
        ("summaries", "summarize_segments", "segment_summaries/shared_s000.json"),
        ("extracts", "extract_segments", "extracts/extract_000.json"),
        ("fragments", "fragment_extracts", "fragments/segment_000/fragment_000.json"),
    ],
)
def test_incomplete_artifacts_regenerate_without_force(
    profile_repo: Path, state: str, stage: str, command: str, relative: str
) -> None:
    config = load_config("mom")
    seed(config)
    path = Path(config["output_root"]) / "shared" / relative
    if state == "missing":
        path.unlink()
    elif state == "truncated":
        path.write_text("{", encoding="utf-8")
    else:
        document = json.loads(path.read_text())
        document["status"] = "partial"
        atomic_json(path, document)
    client = model(stage)
    with (
        patch("jarvis.cli.create_ollama_client", return_value=client),
        patch("jarvis.recovery.create_vector_store", return_value=MagicMock()),
    ):
        assert getattr(cli, f"cmd_{command}")(arguments(config, persist=True), config) == 0
        assert getattr(cli, f"cmd_{command}")(arguments(config, persist=True), config) == 0
    assert json.loads(path.read_text())["status"] == "ok"
    assert client.chat.call_count + client.generate_json.call_count == 1


def test_llm_interruption_reuses_completed_outputs(profile_repo: Path) -> None:
    config = load_config("mom")
    store = seed(config, count=2)
    client = model("extracts")
    client.generate_json.side_effect = [client.generate_json.return_value, KeyboardInterrupt()]
    with (
        patch("jarvis.cli.create_ollama_client", return_value=client),
        patch("jarvis.recovery.create_vector_store", return_value=MagicMock()),
    ):
        with pytest.raises(KeyboardInterrupt):
            cli.cmd_extract_segments(arguments(config, force=True, persist=True), config)
    completed = Path(config["output_root"]) / "shared/extracts/extract_000.json"
    before = completed.read_bytes(), completed.stat().st_mtime_ns
    retry = model("extracts")
    with patch("jarvis.cli.create_ollama_client", return_value=retry):
        assert cli.cmd_extract_segments(arguments(config, persist=True), config) == 0
    assert retry.generate_json.call_count == 1
    assert (completed.read_bytes(), completed.stat().st_mtime_ns) == before
    assert store.count_extracts_for_conversation("shared") == 2


def test_sparse_selection_does_not_delete_missing_input_index(profile_repo: Path) -> None:
    config = load_config("mom")
    store = seed(config, count=3)
    (Path(config["chatgpt_inbox_dir"]) / "shared/segments/segment_001.json").unlink()
    with patch("jarvis.recovery.create_vector_store", return_value=MagicMock()) as vectors:
        CleanupService(config).prepare("extracts", arguments(config, force=True))
    assert store.get_extract_by_segment("shared_s000") is None
    assert store.get_extract_by_segment("shared_s001") is not None
    assert store.get_extract_by_segment("shared_s002") is None
    vectors.return_value.delete_points.assert_called_once_with([str(UUID(int=1)), str(UUID(int=3))])


def test_file_only_force_does_not_initialize_services_or_sqlite(profile_repo: Path) -> None:
    config = load_config("mom")
    config["prompts_dir"] = str(Path(__file__).parents[1] / "prompts")
    source = Path(__file__).parent / "fixtures/chatgpt_tiny.json"
    ingest = argparse.Namespace(source="chatgpt", file=str(source), output_dir=None, persist=False)
    with (
        patch("jarvis.recovery.create_summary_store", side_effect=AssertionError("No SQLite")),
        patch("jarvis.recovery.create_vector_store", side_effect=AssertionError("No Qdrant")),
    ):
        assert cli.cmd_ingest(ingest, config) == 0
        for command, stage in [
            ("extract_segments", "extracts"),
            ("fragment_extracts", "fragments"),
        ]:
            with patch("jarvis.cli.create_ollama_client", return_value=model(stage)):
                assert (
                    getattr(cli, f"cmd_{command}")(
                        arguments(config, conversation_id="conv_test_tiny", force=True), config
                    )
                    == 0
                )
    assert not Path(config["db_path"]).exists()


def test_mom_cli_ingestion_dispatches_and_routes(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parent / "fixtures/chatgpt_tiny.json"
    monkeypatch.setattr(
        sys, "argv", ["jarvis", "--profile", "mom", "ingest", "chatgpt", "--file", str(source)]
    )
    assert cli.main() == 0
    mom = load_config("mom")
    personal = load_config("personal")
    assert (Path(mom["chatgpt_inbox_dir"]) / "conv_test_tiny/normalized.json").exists()
    assert not Path(personal["inbox_root"]).exists()
    assert not Path(mom["db_path"]).exists()


def test_retrieval_uses_selected_profile_and_filters_pending_indexing(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configs = [load_config(profile) for profile in ["personal", "mom"]]
    for config in configs:
        store = seed(config)
        with store._connect() as connection:
            connection.execute(
                "UPDATE fragments SET title = ?", (config["profile_name"] + " title",)
            )
    vector = MagicMock()
    vector.search.return_value = [("shared_s000_x_f000", 0.9, str(UUID(int=1)))]
    embedder = MagicMock()
    embedder.embed.return_value = [1.0, 0.0]
    monkeypatch.setattr(sys, "argv", ["jarvis", "--profile", "mom", "retrieve", "--query", "hello"])
    with (
        patch("jarvis.clients.VectorStore", return_value=vector) as factory,
        patch("jarvis.clients.EmbeddingClient", return_value=embedder),
    ):
        assert cli.main() == 0
    output = capsys.readouterr().out
    assert "mom title" in output and "personal title" not in output
    assert factory.call_args.kwargs["collection_name"] == "jarvis_mom_fragments"
    store = create_summary_store(configs[1])
    store.prepare_fragment_embedding("shared_s000_x_f000", str(UUID(int=1)))
    with (
        patch("jarvis.clients.VectorStore", return_value=vector),
        patch("jarvis.clients.EmbeddingClient", return_value=embedder),
    ):
        assert cli.main() == 0
    assert "No results found" in capsys.readouterr().out


def test_search_refuses_pending_cleanup_before_sqlite_deletion(profile_repo: Path) -> None:
    config = load_config("mom")
    store = seed(config)
    memory = build_memory_layer(config, read_only=True)
    with patch.object(SummaryStore, "invalidate_stage", side_effect=RuntimeError("interrupted")):
        assert cli.cmd_extract_segments(arguments(config, force=True), config) == 1
    assert store.get_extract_by_segment("shared_s000") is not None
    with (
        patch("jarvis.clients.VectorStore", side_effect=AssertionError("No search")),
        patch("jarvis.clients.EmbeddingClient", side_effect=AssertionError("No embedding")),
        pytest.raises(RuntimeError, match="Pending cleanup"),
    ):
        cli.run_retrieval(memory, "query", 5, 1, 0.5)


def test_pending_cleanup_rejects_endpoint_or_range_changes(profile_repo: Path) -> None:
    config = load_config("mom")
    seed(config, count=2)
    with patch("jarvis.recovery.create_vector_store", side_effect=RuntimeError("unavailable")):
        assert cli.cmd_extract_segments(arguments(config, force=True), config) == 1
    directory = CleanupService(config).directory
    journal = next(directory.glob("cleanup_*.json"))
    before = journal.read_bytes()
    with patch("jarvis.recovery.create_vector_store", side_effect=AssertionError("No mutation")):
        changed = {**config, "qdrant_url": "https://synthetic.invalid"}
        with pytest.raises(ValueError, match="selected profile"):
            CleanupService(changed).resume("extracts", "shared")
        with pytest.raises(ValueError, match="different range"):
            CleanupService(config).resume("extracts", "shared", 1, 1)
    assert journal.read_bytes() == before


def test_topic_interruption_reuses_completed_topics(profile_repo: Path) -> None:
    config = load_config("mom")
    store = seed(config, count=3)
    client = model("summaries")
    client.chat.side_effect = [client.chat.return_value, KeyboardInterrupt()]
    embedder = MagicMock()
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    embedder.embed.side_effect = vectors
    with (
        patch("jarvis.cli.create_ollama_client", return_value=client),
        patch("jarvis.cli.create_embedding_client", return_value=embedder),
    ):
        with pytest.raises(KeyboardInterrupt):
            cli.cmd_detect_topics(arguments(config, force=True, persist=True), config)
    first = Path(config["output_root"]) / "shared/topic_summaries/topic_000.json"
    before = first.read_bytes(), first.stat().st_mtime_ns
    first.with_suffix(".md").unlink()
    retry = model("summaries")
    with (
        patch("jarvis.cli.create_ollama_client", return_value=retry),
        patch("jarvis.cli.create_embedding_client", return_value=embedder),
    ):
        for _ in range(2):
            embedder.embed.side_effect = vectors
            assert cli.cmd_detect_topics(arguments(config, persist=True), config) == 0
    assert retry.chat.call_count == 1
    assert (first.read_bytes(), first.stat().st_mtime_ns) == before
    assert first.with_suffix(".md").exists()
    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM topic_summaries").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM topic_segments").fetchone()[0] == 3


def test_valid_extract_repairs_stale_sqlite_without_repeating_model(profile_repo: Path) -> None:
    config = load_config("mom")
    store = seed(config)
    with store._connect() as connection:
        connection.execute("UPDATE extracts SET status = 'partial'")
        connection.execute("UPDATE extract_statements SET text = 'stale' WHERE statement_index = 0")
    path = Path(config["output_root"]) / "shared/extracts/extract_000.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    client = MagicMock()
    with (
        patch("jarvis.cli.create_ollama_client", return_value=client),
        patch("jarvis.recovery.create_vector_store", return_value=MagicMock()) as vectors,
    ):
        assert cli.cmd_extract_segments(arguments(config, persist=True), config) == 0
    client.generate_json.assert_not_called()
    assert store.get_extract_by_segment("shared_s000")["status"] == "ok"
    assert store.get_statements_for_extract("shared_s000_x")[0]["text"] == "hello"
    assert not store.get_fragments_for_conversation("shared")
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    vectors.return_value.delete_points.assert_called_once_with([str(UUID(int=1))])


def test_valid_fragment_batch_removes_stale_extra_records_and_files(profile_repo: Path) -> None:
    config = load_config("mom")
    store = seed(config)
    directory = Path(config["output_root"]) / "shared/fragments/segment_000"
    document = json.loads((directory / "fragment_000.json").read_text())
    atomic_json(directory / "fragment_manifest.json", {"outputs": [document]})
    extra = {**document, "fragment_index": 1}
    atomic_json(directory / "fragment_001.json", extra)
    identifier = MemoryLayer(store).persist_fragment_with_links(extra, "shared_s000_x")
    store.update_fragment_embedding(identifier, str(UUID(int=2)), "synthetic")
    before = (directory / "fragment_000.json").read_bytes()
    client = MagicMock()
    with (
        patch("jarvis.cli.create_ollama_client", return_value=client),
        patch("jarvis.recovery.create_vector_store", return_value=MagicMock()) as vectors,
    ):
        assert cli.cmd_fragment_extracts(arguments(config, persist=True), config) == 0
    client.chat.assert_not_called()
    assert store.get_fragment(identifier) is None
    assert not (directory / "fragment_001.json").exists()
    assert (directory / "fragment_000.json").read_bytes() == before
    assert store.get_link_count("shared_s000_x_f000") == 2
    vectors.return_value.delete_points.assert_called_once_with([str(UUID(int=1)), str(UUID(int=2))])


@pytest.mark.parametrize("name", ["fragment_000.json", "original_export.json"])
def test_force_cleanup_preserves_original_exports(profile_repo: Path, name: str) -> None:
    config = load_config("mom")
    store = seed(config)
    source = Path(config["output_root"]) / "shared/fragments/segment_000" / name
    raw = json.loads((Path(__file__).parent / "fixtures/chatgpt_tiny.json").read_text())
    raw["conversation_id"] = "shared"
    atomic_json(source, raw)
    before = source.read_bytes(), source.stat().st_mtime_ns
    ingest = argparse.Namespace(source="chatgpt", file=str(source), output_dir=None, persist=False)
    assert cli.cmd_ingest(ingest, config) == 0
    with (
        patch("jarvis.cli.create_ollama_client", return_value=model("fragments")),
        patch("jarvis.recovery.create_vector_store", return_value=MagicMock()) as vectors,
    ):
        result = cli.cmd_fragment_extracts(arguments(config, force=True), config)
    if name == "fragment_000.json":
        assert result == 1
        vectors.assert_not_called()
        assert store.get_fragment("shared_s000_x_f000") is not None
    else:
        assert result == 0
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


@pytest.mark.parametrize("checkpoint_state", ["valid", "truncated", "invalid"])
def test_fragment_batch_recovery_reuses_only_valid_completed_batches(
    profile_repo: Path, checkpoint_state: str
) -> None:
    config = load_config("mom")
    seed(config)
    path = Path(config["output_root"]) / "shared/extracts/extract_000.json"
    document = json.loads(path.read_text())
    document["statements"] = [
        {"statement_index": index, "speaker": "user", "text": f"Statement {index}"}
        for index in range(_MAX_STATEMENTS_PER_BATCH + 1)
    ]
    atomic_json(path, document)
    calls = []

    def fragment_batch(self: Fragmenter, statements: list, **kwargs: Any) -> tuple:
        calls.append(statements)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return ([{"title": "Batch", "statements": statements}], 1, False, False, None, "")

    with (
        patch("jarvis.cli.create_ollama_client", return_value=model("fragments")),
        patch("jarvis.recovery.create_vector_store", return_value=MagicMock()),
        patch.object(Fragmenter, "_run_fragment_llm", autospec=True, side_effect=fragment_batch),
    ):
        with pytest.raises(KeyboardInterrupt):
            cli.cmd_fragment_extracts(arguments(config, force=True), config)
    directory = Path(config["output_root"]) / "shared/fragments/segment_000"
    checkpoint = directory / ".batch_000.json"
    assert checkpoint.exists()
    if checkpoint_state == "truncated":
        checkpoint.write_text("{", encoding="utf-8")
    elif checkpoint_state == "invalid":
        cached = json.loads(checkpoint.read_text())
        cached["result"] = ["invalid"]
        atomic_json(checkpoint, cached)

    def retry_batch(self: Fragmenter, statements: list, **kwargs: Any) -> tuple:
        return ([{"title": "Batch", "statements": statements}], 1, False, False, None, "")

    with (
        patch("jarvis.cli.create_ollama_client", return_value=model("fragments")),
        patch.object(
            Fragmenter, "_run_fragment_llm", autospec=True, side_effect=retry_batch
        ) as retry,
    ):
        assert cli.cmd_fragment_extracts(arguments(config), config) == 0
    assert retry.call_count == (1 if checkpoint_state == "valid" else 2)
    assert len(fragment_outputs(directory, document)) == 2


@pytest.mark.parametrize("content", ["{", '{"items": "invalid"}'])
def test_corrupt_cleanup_journal_is_retained(profile_repo: Path, content: str) -> None:
    config = load_config("mom")
    seed(config)
    with patch("jarvis.recovery.create_vector_store", side_effect=RuntimeError("unavailable")):
        assert cli.cmd_extract_segments(arguments(config, force=True), config) == 1
    journal = next(CleanupService(config).directory.glob("cleanup_*.json"))
    journal.write_text(content, encoding="utf-8")
    with patch("jarvis.recovery.create_vector_store", side_effect=AssertionError("No mutation")):
        assert cli.cmd_extract_segments(arguments(config), config) == 1
    assert journal.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("command", ["summarize_segments", "detect_topics"])
def test_incomplete_model_summary_is_not_persisted(profile_repo: Path, command: str) -> None:
    config = load_config("mom")
    seed(config)
    client = model("summaries")
    client.parse_json_response.return_value = ({}, False, "")
    client.parse_json_response.side_effect = None
    embedder = MagicMock()
    embedder.embed.return_value = [1.0, 0.0]
    with (
        patch("jarvis.cli.create_ollama_client", return_value=client),
        patch("jarvis.cli.create_embedding_client", return_value=embedder),
        patch("jarvis.cli._build_memory_layer") as persistence,
    ):
        assert (
            getattr(cli, f"cmd_{command}")(arguments(config, force=True, persist=True), config) == 1
        )
    persistence.assert_not_called()


@pytest.fixture
def scope_profiles(profile_repo: Path) -> tuple[dict, dict]:
    active, other = load_config("mom"), load_config("personal")
    for config in (active, other):
        store = seed(config, count=3)
        seed(config, "unrelated")
        gc.collect()
        with closing(store._connect()) as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
    return active, other


def scope_snapshot(root: Path) -> dict[Path, tuple[bytes | None, int]]:
    return {
        path: (path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns)
        for path in root.rglob("*")
    }


@contextmanager
def forbid_cleanup_mutations() -> Iterator[None]:
    with (
        patch(
            "jarvis.recovery.atomic_json", side_effect=AssertionError("Journal write")
        ) as journal,
        patch.object(
            SummaryStore, "invalidate_stage", side_effect=AssertionError("SQLite mutation")
        ) as database,
        patch(
            "jarvis.recovery.create_vector_store", side_effect=AssertionError("Qdrant access")
        ) as vectors,
        patch.object(Path, "unlink", side_effect=AssertionError("File deletion")) as unlink,
        patch.object(Path, "mkdir", side_effect=AssertionError("Directory creation")) as mkdir,
        patch(
            "jarvis.cli.create_ollama_client", side_effect=AssertionError("Inference")
        ) as inference,
    ):
        yield
        for operation in (journal, database, vectors, unlink, mkdir, inference):
            operation.assert_not_called()


def scope_input(config: dict, stage: str, index: int) -> Path:
    if stage == "fragment_extracts":
        return Path(config["output_root"]) / "shared/extracts" / f"extract_{index:03d}.json"
    return Path(config["chatgpt_inbox_dir"]) / "shared/segments" / f"segment_{index:03d}.json"


@pytest.mark.parametrize("command", ["summarize_segments", "extract_segments", "fragment_extracts"])
@pytest.mark.parametrize("swapped", ["segment_id", "segment_index", "records"])
def test_ranged_cleanup_rejects_swapped_artifact_scope_before_mutation(
    profile_repo: Path, scope_profiles: tuple[dict, dict], command: str, swapped: str
) -> None:
    config, _ = scope_profiles
    first, second = [scope_input(config, command, index) for index in (1, 2)]
    left, right = [json.loads(path.read_text()) for path in (first, second)]
    if swapped == "records":
        left, right = right, left
    else:
        left[swapped], right[swapped] = right[swapped], left[swapped]
    atomic_json(first, left)
    atomic_json(second, right)
    before = scope_snapshot(profile_repo)
    with forbid_cleanup_mutations():
        assert (
            getattr(cli, f"cmd_{command}")(
                arguments(config, force=True, from_segment=1, to_segment=1), config
            )
            == 1
        )
    assert scope_snapshot(profile_repo) == before


def remove_scope_segment(store: SummaryStore) -> None:
    with closing(store._connect()) as connection, connection:
        connection.execute("DELETE FROM fragments WHERE extract_id = 'shared_s001_x'")
        connection.execute("DELETE FROM extracts WHERE segment_id = 'shared_s001'")
        connection.execute("DELETE FROM segment_summaries WHERE segment_id = 'shared_s001'")
        connection.execute("DELETE FROM topic_segments WHERE segment_id = 'shared_s001'")
        connection.execute("DELETE FROM segments WHERE segment_id = 'shared_s001'")


def swap_scope_indices(store: SummaryStore) -> None:
    with closing(store._connect()) as connection, connection:
        connection.execute(
            "UPDATE segments SET segment_index = 99 WHERE segment_id = 'shared_s001'"
        )
        connection.execute("UPDATE segments SET segment_index = 1 WHERE segment_id = 'shared_s002'")
        connection.execute("UPDATE segments SET segment_index = 2 WHERE segment_id = 'shared_s001'")


@pytest.mark.parametrize(
    "command, mismatch",
    [
        (command, mismatch)
        for command in ("summarize_segments", "extract_segments", "fragment_extracts")
        for mismatch in ("segment_index", "conversation", "slot_id")
    ]
    + [("fragment_extracts", "extract_index"), ("fragment_extracts", "extract_conversation")],
)
def test_ranged_cleanup_rejects_sqlite_scope_mismatches(
    profile_repo: Path, scope_profiles: tuple[dict, dict], command: str, mismatch: str
) -> None:
    config, _ = scope_profiles
    store = create_summary_store(config)
    if mismatch == "segment_index":
        swap_scope_indices(store)
    elif mismatch == "slot_id":
        row = store.get_segment("shared_s001")
        remove_scope_segment(store)
        row["segment_id"] = "different_identifier"
        store.insert_segment(row)
    else:
        with closing(store._connect()) as connection, connection:
            if mismatch == "conversation":
                connection.execute(
                    "UPDATE segments SET conversation_id = ? WHERE segment_id = ?",
                    ("unrelated", "shared_s001"),
                )
            elif mismatch == "extract_index":
                connection.execute(
                    "UPDATE extracts SET segment_index = 2 WHERE segment_id = 'shared_s001'"
                )
            else:
                connection.execute(
                    "UPDATE extracts SET parent_conversation_id = ? WHERE segment_id = ?",
                    ("unrelated", "shared_s001"),
                )
    before = scope_snapshot(profile_repo)
    with forbid_cleanup_mutations():
        assert (
            getattr(cli, f"cmd_{command}")(
                arguments(config, force=True, from_segment=1, to_segment=1), config
            )
            == 1
        )
    assert scope_snapshot(profile_repo) == before


@pytest.mark.parametrize("stage", ["summaries", "extracts", "fragments"])
@pytest.mark.parametrize("mismatch", ["journal_id", "sqlite_index"])
def test_saved_cleanup_scope_is_revalidated_before_replay(
    profile_repo: Path, scope_profiles: tuple[dict, dict], stage: str, mismatch: str
) -> None:
    config, _ = scope_profiles
    item = {"segment_id": "shared_s001", "segment_index": 1}
    service = CleanupService(config)
    with patch.object(SummaryStore, "invalidate_stage", side_effect=RuntimeError("Interrupted")):
        with pytest.raises(RuntimeError, match="Interrupted"):
            service.invalidate(stage, "shared", [item])
    journal = service._journal(stage, "shared")
    if mismatch == "journal_id":
        plan = json.loads(journal.read_text())
        plan["items"][0]["segment_id"] = "shared_s002"
        atomic_json(journal, plan)
    else:
        swap_scope_indices(create_summary_store(config))
    before = scope_snapshot(profile_repo)
    with forbid_cleanup_mutations(), pytest.raises(ValueError):
        service.resume(stage, "shared", 1, 1)
    assert scope_snapshot(profile_repo) == before


@pytest.mark.parametrize(
    "stage, command",
    [
        ("summaries", "summarize_segments"),
        ("extracts", "extract_segments"),
        ("fragments", "fragment_extracts"),
    ],
)
def test_pending_cleanup_does_not_precede_live_input_validation(
    profile_repo: Path, scope_profiles: tuple[dict, dict], stage: str, command: str
) -> None:
    config, _ = scope_profiles
    with patch.object(SummaryStore, "invalidate_stage", side_effect=RuntimeError("Interrupted")):
        with pytest.raises(RuntimeError, match="Interrupted"):
            CleanupService(config).invalidate(
                stage, "shared", [{"segment_id": "shared_s001", "segment_index": 1}]
            )
    source = scope_input(config, command, 1)
    data = json.loads(source.read_text())
    data["segment_id"] = "shared_s002"
    atomic_json(source, data)
    before = scope_snapshot(profile_repo)
    with forbid_cleanup_mutations():
        assert (
            getattr(cli, f"cmd_{command}")(arguments(config, from_segment=1, to_segment=1), config)
            == 1
        )
    assert scope_snapshot(profile_repo) == before


@pytest.mark.parametrize(
    "stage, command",
    [
        ("summaries", "summarize_segments"),
        ("extracts", "extract_segments"),
        ("fragments", "fragment_extracts"),
    ],
)
def test_journal_replay_allows_already_removed_records_and_inputs(
    scope_profiles: tuple[dict, dict], stage: str, command: str
) -> None:
    config, other = scope_profiles
    store = create_summary_store(config)
    original = SummaryStore.invalidate_stage

    def interrupted(self: SummaryStore, *args: Any) -> None:
        original(self, *args)
        raise RuntimeError("Interrupted after SQLite cleanup")

    service = CleanupService(config)
    with patch.object(SummaryStore, "invalidate_stage", interrupted):
        with pytest.raises(RuntimeError, match="Interrupted after"):
            service.invalidate(stage, "shared", [{"segment_id": "shared_s001", "segment_index": 1}])
    scope_input(config, command, 1).unlink()
    remove_scope_segment(store)
    other_before = scope_snapshot(Path(other["output_root"]))
    outside = store.get_fragment("shared_s002_x_f000")
    with patch("jarvis.recovery.create_vector_store", return_value=MagicMock()) as vectors:
        service.resume(stage, "shared", 1, 1)
    assert not service._journal(stage, "shared").exists()
    if stage == "summaries":
        vectors.assert_not_called()
    else:
        vectors.return_value.delete_points.assert_called_once_with([str(UUID(int=2))])
    assert store.get_fragment("shared_s002_x_f000") == outside
    assert scope_snapshot(Path(other["output_root"])) == other_before


@pytest.mark.parametrize(
    "stage, command",
    [
        ("summaries", "summarize_segments"),
        ("extracts", "extract_segments"),
        ("fragments", "fragment_extracts"),
    ],
)
def test_file_only_scope_remains_valid_without_persisted_segment(
    scope_profiles: tuple[dict, dict], stage: str, command: str
) -> None:
    config, _ = scope_profiles
    remove_scope_segment(create_summary_store(config))
    with (
        patch(
            "jarvis.recovery.create_vector_store", side_effect=AssertionError("No vectors")
        ) as vectors,
        patch("jarvis.cli.create_ollama_client", return_value=model(stage)),
    ):
        assert (
            getattr(cli, f"cmd_{command}")(
                arguments(config, force=True, from_segment=1, to_segment=1), config
            )
            == 0
        )
    vectors.assert_not_called()
