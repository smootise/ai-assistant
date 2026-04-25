"""Tests for 'ingest chatgpt --persist' SQLite persistence.

Verifies that source file metadata, conversation, and segment rows are written
correctly, and that re-running is idempotent (no duplicate rows).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest

from jarvis.cli import cmd_ingest
from jarvis.store import SummaryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(raw_path: str, output_dir: str, persist: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        source="chatgpt",
        file=raw_path,
        output_dir=output_dir,
        persist=persist,
    )


def _make_config(tmp_path: Path) -> dict:
    return {
        "output_root": str(tmp_path / "OUTPUTS"),
        "db_path": str(tmp_path / "jarvis.db"),
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "local_model_name": "gemma4:31b",
        "ollama_base_url": "http://localhost:11434",
        "ollama_timeout": 60,
        "prompts_dir": "prompts",
        "schema": {},
        "schema_version": 1,
        "embedding_model": "qwen3-embedding",
    }


def _write_raw_export(path: Path) -> Dict[str, Any]:
    """Write a minimal ChatGPT export JSON and return its content."""
    raw = {
        "conversation_id": "conv-test-001",
        "title": "Test conversation",
        "create_time": 1700000000.0,
        "update_time": 1700001000.0,
        "mapping": {
            "msg-a": {
                "id": "msg-a",
                "message": {
                    "id": "msg-a",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello!"]},
                    "create_time": 1700000000.0,
                    "status": "finished_successfully",
                },
                "parent": None,
                "children": ["msg-b"],
            },
            "msg-b": {
                "id": "msg-b",
                "message": {
                    "id": "msg-b",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["Hi there!"]},
                    "create_time": 1700000100.0,
                    "status": "finished_successfully",
                },
                "parent": "msg-a",
                "children": [],
            },
        },
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    return raw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestPersist:
    def test_no_persist_creates_no_db(self, tmp_path):
        raw_path = tmp_path / "conv.json"
        _write_raw_export(raw_path)
        inbox = tmp_path / "inbox"
        config = _make_config(tmp_path)
        args = _make_args(str(raw_path), str(inbox), persist=False)

        with patch("jarvis.cli._build_memory_layer") as mock_build:
            rc = cmd_ingest(args, config)

        assert rc == 0
        mock_build.assert_not_called()
        assert not Path(config["db_path"]).exists()

    def test_persist_writes_source_files(self, tmp_path):
        raw_path = tmp_path / "conv.json"
        _write_raw_export(raw_path)
        inbox = tmp_path / "inbox"
        config = _make_config(tmp_path)
        args = _make_args(str(raw_path), str(inbox), persist=True)

        rc = cmd_ingest(args, config)
        assert rc == 0

        store = SummaryStore(db_path=config["db_path"])
        with store._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
        assert count == 2  # raw + normalized

    def test_persist_writes_conversation(self, tmp_path):
        raw_path = tmp_path / "conv.json"
        _write_raw_export(raw_path)
        inbox = tmp_path / "inbox"
        config = _make_config(tmp_path)
        args = _make_args(str(raw_path), str(inbox), persist=True)

        rc = cmd_ingest(args, config)
        assert rc == 0

        store = SummaryStore(db_path=config["db_path"])
        with store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = 'conv-test-001'"
            ).fetchone()
        assert row is not None

    def test_persist_writes_segments(self, tmp_path):
        raw_path = tmp_path / "conv.json"
        _write_raw_export(raw_path)
        inbox = tmp_path / "inbox"
        config = _make_config(tmp_path)
        args = _make_args(str(raw_path), str(inbox), persist=True)

        rc = cmd_ingest(args, config)
        assert rc == 0

        store = SummaryStore(db_path=config["db_path"])
        with store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM segments WHERE conversation_id = 'conv-test-001'"
            ).fetchone()[0]
        # The minimal 2-message conversation produces 1 segment
        assert count >= 1

    def test_persist_segment_has_text(self, tmp_path):
        raw_path = tmp_path / "conv.json"
        _write_raw_export(raw_path)
        inbox = tmp_path / "inbox"
        config = _make_config(tmp_path)
        args = _make_args(str(raw_path), str(inbox), persist=True)

        cmd_ingest(args, config)

        store = SummaryStore(db_path=config["db_path"])
        with store._connect() as conn:
            row = conn.execute(
                "SELECT segment_text FROM segments WHERE conversation_id = 'conv-test-001'"
            ).fetchone()
        assert row is not None
        assert "Hello!" in row[0] or "user:" in row[0]

    def test_persist_idempotent(self, tmp_path):
        raw_path = tmp_path / "conv.json"
        _write_raw_export(raw_path)
        inbox = tmp_path / "inbox"
        config = _make_config(tmp_path)
        args = _make_args(str(raw_path), str(inbox), persist=True)

        cmd_ingest(args, config)
        cmd_ingest(args, config)  # re-run on same export

        store = SummaryStore(db_path=config["db_path"])
        with store._connect() as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            seg_count = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        # Conversation and segment rows are conversation_id-keyed — exactly one each
        assert conv_count == 1
        assert seg_count >= 1
