"""Tests for fragment-extracts CLI flag semantics.

Covers the four cases:
1. no flags       — disk artifacts only; no SQLite, no Qdrant
2. --persist      — SQLite only; no Qdrant
3. --persist --embed — SQLite then Qdrant
4. --embed alone  — fail fast with exit code 2
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from jarvis.cli import cmd_fragment_extracts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output_data(fragment_index: int, conv_id: str = "conv-001") -> Dict[str, Any]:
    return {
        "source_kind": "ai_chat_fragment",
        "source_file": f"fragment_{fragment_index:03d}.json",
        "parent_conversation_id": conv_id,
        "fragment_index": fragment_index,
        "fragment_title": f"Fragment {fragment_index}",
        "statements": [],
        "status": "ok",
        "model": "gemma4:31b",
        "created_at": "2026-04-23T00:00:00Z",
    }


def _make_args(
    *,
    persist: bool = False,
    embed: bool = False,
    force: bool = False,
    conv_id: str = "conv-001",
) -> argparse.Namespace:
    return argparse.Namespace(
        source="chatgpt",
        conversation_id=conv_id,
        from_segment=None,
        to_segment=None,
        persist=persist,
        embed=embed,
        force=force,
        retries=1,
    )


def _make_config(tmp_path: Path) -> dict:
    (tmp_path / "conv-001" / "extracts").mkdir(parents=True)
    return {
        "output_root": str(tmp_path),
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


def _canned_results(conv_id: str = "conv-001") -> Tuple[List, List]:
    """Two fragment results, zero skipped segments."""
    out_dir = Path("/fake/fragments")
    return (
        [
            (out_dir, _make_output_data(0, conv_id)),
            (out_dir, _make_output_data(1, conv_id)),
        ],
        [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoFlags:
    """No --persist, no --embed → disk artifacts only."""

    def test_no_memory_layer_built(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=False, embed=False)
        results, skipped = _canned_results()

        with (
            patch("jarvis.cli.Fragmenter") as MockFragmenter,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockFragmenter.return_value.fragment_conversation_extracts.return_value = (
                results, skipped
            )
            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        mock_build.assert_not_called()

    def test_persist_sqlite_not_called(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=False, embed=False)
        results, skipped = _canned_results()

        with (
            patch("jarvis.cli.Fragmenter") as MockFragmenter,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockFragmenter.return_value.fragment_conversation_extracts.return_value = (
                results, skipped
            )
            mock_memory = MagicMock()
            mock_build.return_value = mock_memory
            cmd_fragment_extracts(args, config)

        mock_memory.persist_sqlite.assert_not_called()
        mock_memory.index_in_qdrant.assert_not_called()


class TestPersistOnly:
    """--persist only → SQLite written; Qdrant not touched."""

    def test_persist_sqlite_called_per_fragment(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=True, embed=False)
        results, skipped = _canned_results()

        with (
            patch("jarvis.cli.Fragmenter") as MockFragmenter,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockFragmenter.return_value.fragment_conversation_extracts.return_value = (
                results, skipped
            )
            mock_memory = MagicMock()
            mock_memory.store.get_by_source_file.return_value = None
            mock_memory.persist_sqlite.side_effect = [10, 11]
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        assert mock_memory.persist_sqlite.call_count == len(results)
        mock_memory.index_in_qdrant.assert_not_called()

    def test_already_persisted_fragment_skipped(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=True, embed=False)
        results, skipped = _canned_results()

        with (
            patch("jarvis.cli.Fragmenter") as MockFragmenter,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockFragmenter.return_value.fragment_conversation_extracts.return_value = (
                results, skipped
            )
            mock_memory = MagicMock()
            mock_memory.store.get_by_source_file.return_value = {
                "summary_id": 99, "qdrant_point_id": "abc-123"
            }
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        mock_memory.persist_sqlite.assert_not_called()
        mock_memory.index_in_qdrant.assert_not_called()


class TestPersistAndEmbed:
    """--persist --embed → SQLite then Qdrant for each fragment."""

    def test_both_called_in_order(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=True, embed=True)
        results, skipped = _canned_results()
        summary_ids = [10, 11]

        with (
            patch("jarvis.cli.Fragmenter") as MockFragmenter,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockFragmenter.return_value.fragment_conversation_extracts.return_value = (
                results, skipped
            )
            mock_memory = MagicMock()
            mock_memory.store.get_by_source_file.return_value = None
            mock_memory.persist_sqlite.side_effect = summary_ids
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        assert mock_memory.persist_sqlite.call_count == len(results)
        assert mock_memory.index_in_qdrant.call_count == len(results)

        # Verify ordering: index_in_qdrant received the summary_id from persist_sqlite
        _, out0 = results[0]
        _, out1 = results[1]
        mock_memory.index_in_qdrant.assert_any_call(summary_id=10, output_data=out0)
        mock_memory.index_in_qdrant.assert_any_call(summary_id=11, output_data=out1)

    def test_sqlite_only_then_embed_indexes_existing_rows(self, tmp_path):
        """--persist --embed on rows already in SQLite but not Qdrant → index only."""
        config = _make_config(tmp_path)
        args = _make_args(persist=True, embed=True)
        results, skipped = _canned_results()

        with (
            patch("jarvis.cli.Fragmenter") as MockFragmenter,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockFragmenter.return_value.fragment_conversation_extracts.return_value = (
                results, skipped
            )
            mock_memory = MagicMock()
            # Already in SQLite, not yet in Qdrant
            mock_memory.store.get_by_source_file.return_value = {
                "summary_id": 99, "qdrant_point_id": None
            }
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        mock_memory.persist_sqlite.assert_not_called()
        assert mock_memory.index_in_qdrant.call_count == len(results)
        mock_memory.index_in_qdrant.assert_any_call(summary_id=99, output_data=results[0][1])
        mock_memory.index_in_qdrant.assert_any_call(summary_id=99, output_data=results[1][1])

    def test_already_fully_indexed_skipped_on_embed(self, tmp_path):
        """--persist --embed on rows already in SQLite AND Qdrant → skip entirely."""
        config = _make_config(tmp_path)
        args = _make_args(persist=True, embed=True)
        results, skipped = _canned_results()

        with (
            patch("jarvis.cli.Fragmenter") as MockFragmenter,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockFragmenter.return_value.fragment_conversation_extracts.return_value = (
                results, skipped
            )
            mock_memory = MagicMock()
            # Already fully persisted and indexed
            mock_memory.store.get_by_source_file.return_value = {
                "summary_id": 99, "qdrant_point_id": "abc-123"
            }
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        mock_memory.persist_sqlite.assert_not_called()
        mock_memory.index_in_qdrant.assert_not_called()


class TestEmbedWithoutPersist:
    """--embed without --persist → exit code 2, no side effects."""

    def test_fails_with_exit_code_2(self, tmp_path, capsys):
        test_argv = [
            "jarvis.cli",
            "fragment-extracts", "chatgpt",
            "--conversation-id", "conv-001",
            "--embed",
        ]
        with (
            patch.object(sys, "argv", test_argv),
            patch("jarvis.cli._build_memory_layer") as mock_build,
            pytest.raises(SystemExit) as exc_info,
        ):
            from jarvis.cli import main
            main()

        assert exc_info.value.code == 2
        mock_build.assert_not_called()

    def test_error_message_mentions_persist(self, tmp_path, capsys):
        test_argv = [
            "jarvis.cli",
            "fragment-extracts", "chatgpt",
            "--conversation-id", "conv-001",
            "--embed",
        ]
        with (
            patch.object(sys, "argv", test_argv),
            pytest.raises(SystemExit),
        ):
            from jarvis.cli import main
            main()

        captured = capsys.readouterr()
        assert "--embed requires --persist" in captured.err
