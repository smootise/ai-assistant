"""Tests for extract-segments CLI flag semantics.

Covers:
1. no --persist  → disk artifacts only; no SQLite writes
2. --persist     → extracts and statements written to SQLite; idempotent
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from jarvis.cli import cmd_extract_segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output_data(
    seg_idx: int = 0,
    conv_id: str = "conv-001",
    from_disk: bool = False,
) -> Dict[str, Any]:
    seg_id = f"{conv_id}_s{seg_idx:03d}"
    data = {
        "source_kind": "ai_chat_extract",
        "source_file": f"extract_{seg_idx:03d}.json",
        "segment_id": seg_id,
        "segment_index": seg_idx,
        "parent_conversation_id": conv_id,
        "provider": "local",
        "model": "gemma4:31b",
        "status": "ok",
        "statements": [
            {
                "statement_id": f"{seg_id}_st0000",
                "statement_index": 0,
                "segment_id": seg_id,
                "segment_index": seg_idx,
                "parent_conversation_id": conv_id,
                "speaker": "user",
                "text": "hello",
            },
            {
                "statement_id": f"{seg_id}_st0001",
                "statement_index": 1,
                "segment_id": seg_id,
                "segment_index": seg_idx,
                "parent_conversation_id": conv_id,
                "speaker": "assistant",
                "text": "hi",
            },
        ],
        "created_at": "2026-04-23T00:00:00Z",
    }
    if from_disk:
        data["_from_disk"] = True
    return data


def _make_args(
    *,
    persist: bool = False,
    force: bool = False,
    conv_id: str = "conv-001",
    inbox_dir: str = "inbox/ai_chat/chatgpt",
) -> argparse.Namespace:
    return argparse.Namespace(
        source="chatgpt",
        conversation_id=conv_id,
        inbox_dir=inbox_dir,
        from_segment=None,
        to_segment=None,
        persist=persist,
        force=force,
        retries=1,
    )


def _make_config(tmp_path: Path) -> dict:
    inbox_dir = tmp_path / "inbox" / "ai_chat" / "chatgpt"
    segments_dir = inbox_dir / "conv-001" / "segments"
    segments_dir.mkdir(parents=True)
    return {
        "inbox_dir": str(inbox_dir),
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


def _canned_results(conv_id: str = "conv-001") -> List[Tuple[Path, Dict[str, Any]]]:
    out_dir = Path("/fake/extracts")
    return [
        (out_dir, _make_output_data(0, conv_id)),
        (out_dir, _make_output_data(1, conv_id)),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoFlags:
    """No --persist → disk artifacts only."""

    def test_no_memory_layer_built(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=False, inbox_dir=config["inbox_dir"])
        results = _canned_results()

        with (
            patch("jarvis.cli.SegmentExtractor") as MockExtractor,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockExtractor.return_value.extract_conversation_segments.return_value = results
            rc = cmd_extract_segments(args, config)

        assert rc == 0
        mock_build.assert_not_called()

    def test_no_sqlite_writes(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=False, inbox_dir=config["inbox_dir"])
        results = _canned_results()

        with (
            patch("jarvis.cli.SegmentExtractor") as MockExtractor,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockExtractor.return_value.extract_conversation_segments.return_value = results
            mock_memory = MagicMock()
            mock_build.return_value = mock_memory
            cmd_extract_segments(args, config)

        mock_memory.persist_extract_with_statements.assert_not_called()


class TestPersistFlag:
    """--persist → extract + statement rows written; idempotent."""

    def test_persist_called_per_extract(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=True, inbox_dir=config["inbox_dir"])
        results = _canned_results()

        with (
            patch("jarvis.cli.SegmentExtractor") as MockExtractor,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockExtractor.return_value.extract_conversation_segments.return_value = results
            mock_memory = MagicMock()
            mock_memory.store.get_extract_by_segment.return_value = None
            mock_build.return_value = mock_memory

            rc = cmd_extract_segments(args, config)

        assert rc == 0
        assert mock_memory.persist_extract_with_statements.call_count == len(results)

    def test_already_persisted_extract_skipped(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=True, inbox_dir=config["inbox_dir"])
        results = _canned_results()

        with (
            patch("jarvis.cli.SegmentExtractor") as MockExtractor,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockExtractor.return_value.extract_conversation_segments.return_value = results
            mock_memory = MagicMock()
            mock_memory.store.get_extract_by_segment.return_value = {
                "extract_id": "conv-001_s000_x"
            }
            mock_build.return_value = mock_memory

            rc = cmd_extract_segments(args, config)

        assert rc == 0
        mock_memory.persist_extract_with_statements.assert_not_called()

    def test_persist_from_disk_results(self, tmp_path):
        """Resume path: results loaded from disk are persisted to SQLite."""
        config = _make_config(tmp_path)
        args = _make_args(persist=True, inbox_dir=config["inbox_dir"])
        results = [
            (Path("/fake"), _make_output_data(0, from_disk=True)),
        ]

        with (
            patch("jarvis.cli.SegmentExtractor") as MockExtractor,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockExtractor.return_value.extract_conversation_segments.return_value = results
            mock_memory = MagicMock()
            mock_memory.store.get_extract_by_segment.return_value = None
            mock_build.return_value = mock_memory

            rc = cmd_extract_segments(args, config)

        assert rc == 0
        # From-disk result must still be persisted (no LLM call happened, but data is valid)
        mock_memory.persist_extract_with_statements.assert_called_once()
