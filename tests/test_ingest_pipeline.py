"""Tests for the ingest_chatgpt pipeline function."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.ingest.pipeline import ingest_chatgpt, sha256_file, source_file_data


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
        config={},
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
            config={},
        )


def test_ingest_chatgpt_persist(tmp_path):
    config = {
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
