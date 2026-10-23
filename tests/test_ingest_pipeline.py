"""Tests for the ingest_chatgpt pipeline function."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.config import load_config
from jarvis.ingest import pipeline
from jarvis.ingest.pipeline import ingest_chatgpt, sha256_file, source_file_data
from jarvis.paths import PathBoundaryError

FIXTURE = Path(__file__).parent / "fixtures" / "chatgpt_tiny.json"


def test_sha256_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello")
    result = sha256_file(f)
    assert len(result) == 64
    assert result == sha256_file(f)


def test_source_file_data(tmp_path):
    f = tmp_path / "export.json"
    f.write_bytes(b"content")
    data = source_file_data(f, "chatgpt_raw_export")
    assert data["source_kind"] == "chatgpt_raw_export"
    assert data["original_filename"] == "export.json"
    assert data["sha256"] == data["source_file_id"]
    assert data["size_bytes"] == len(b"content")


def test_ingest_chatgpt_no_persist(tmp_path):
    result = ingest_chatgpt(
        raw_path=FIXTURE,
        output_dir=tmp_path,
        persist=False,
        config={"inbox_root": str(tmp_path)},
    )
    assert result["conversation_id"] == "conv_test_tiny"
    assert result["segment_count"] >= 1
    norm_path = Path(result["normalized_path"])
    assert norm_path.exists()
    normalized = json.loads(norm_path.read_text())
    assert normalized["conversation_id"] == "conv_test_tiny"
    seg_dir = tmp_path / "conv_test_tiny" / "segments"
    assert seg_dir.exists()


def test_ingest_chatgpt_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_chatgpt(
            raw_path=tmp_path / "nonexistent.json",
            output_dir=tmp_path,
            persist=False,
            config={"inbox_root": str(tmp_path)},
        )


@pytest.mark.parametrize("profile", ["personal", "mom"])
@pytest.mark.parametrize("persist", [False, True])
def test_ingest_rejects_source_at_normalized_staging_path_before_writing(
    profile_repo: Path, profile: str, persist: bool
) -> None:
    config = load_config(profile)
    inbox = Path(config["chatgpt_inbox_dir"])
    conversation_dir = inbox / "conv_test_tiny"
    conversation_dir.mkdir(parents=True)
    source = conversation_dir / "normalized.tmp"
    original = FIXTURE.read_bytes()
    source.write_bytes(original)
    original_mtime = source.stat().st_mtime_ns

    with (
        patch.object(
            pipeline, "save_normalized", wraps=pipeline.save_normalized
        ) as save_normalized,
        patch.object(pipeline, "save_segments", wraps=pipeline.save_segments) as save_segments,
        patch("jarvis.clients.create_summary_store") as create_store,
    ):
        with pytest.raises(PathBoundaryError, match="Source overlaps generated artifacts"):
            ingest_chatgpt(source, inbox, persist, config)

        save_normalized.assert_not_called()
        save_segments.assert_not_called()
        create_store.assert_not_called()

    assert source.read_bytes() == original
    assert source.stat().st_mtime_ns == original_mtime
    assert list(conversation_dir.iterdir()) == [source]
    assert not Path(config["db_path"]).exists()


def test_ingest_chatgpt_persist(tmp_path):
    config = {
        "inbox_root": str(tmp_path),
        "db_path": str(tmp_path / "test.db"),
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "embedding_model": "test-embed",
        "ollama_base_url": "http://localhost:11434",
    }

    mock_memory = MagicMock()
    result = ingest_chatgpt(
        raw_path=FIXTURE,
        output_dir=tmp_path,
        persist=True,
        config=config,
        memory=mock_memory,
    )

    assert result["conversation_id"] == "conv_test_tiny"
    assert result["segment_count"] >= 1

    mock_memory.persist_source_file.assert_called()
    mock_memory.persist_conversation.assert_called_once()
    assert mock_memory.persist_segment.call_count == result["segment_count"]

    call_args = mock_memory.persist_conversation.call_args[0][0]
    assert call_args["conversation_id"] == "conv_test_tiny"
