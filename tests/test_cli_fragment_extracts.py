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

def _make_output_data(
    fragment_index: int,
    conv_id: str = "conv-001",
    seg_idx: int = 0,
) -> Dict[str, Any]:
    seg_id = f"{conv_id}_s{seg_idx:03d}"
    return {
        "source_kind": "ai_chat_fragment",
        "source_file": f"segment_{seg_idx:03d}/fragment_{fragment_index:03d}.json",
        "parent_conversation_id": conv_id,
        "segment_id": seg_id,
        "segment_index": seg_idx,
        "fragment_index": fragment_index,
        "title": f"Fragment {fragment_index}",
        "text": "user: hello\n\nassistant: hi",
        "statements": [
            {"statement_index": 0, "speaker": "user", "text": "hello"},
        ],
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

    def test_no_persist_calls(self, tmp_path):
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

        mock_memory.persist_fragment_with_links.assert_not_called()
        mock_memory.index_fragment_in_qdrant.assert_not_called()


class TestPersistOnly:
    """--persist only → SQLite written; Qdrant not touched."""

    def test_persist_fragment_called_per_fragment(self, tmp_path):
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
            mock_memory.store.get_fragment.return_value = None
            mock_memory.persist_fragment_with_links.side_effect = [
                "conv-001_s000_x_f000",
                "conv-001_s000_x_f001",
            ]
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        assert mock_memory.persist_fragment_with_links.call_count == len(results)
        mock_memory.index_fragment_in_qdrant.assert_not_called()

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
            # Already persisted and embedded
            mock_memory.store.get_fragment.return_value = {
                "fragment_id": "conv-001_s000_x_f000",
                "qdrant_point_id": "abc-123",
            }
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        mock_memory.persist_fragment_with_links.assert_not_called()
        mock_memory.index_fragment_in_qdrant.assert_not_called()


class TestPersistAndEmbed:
    """--persist --embed → SQLite then Qdrant for each fragment."""

    def test_both_called_in_order(self, tmp_path):
        config = _make_config(tmp_path)
        args = _make_args(persist=True, embed=True)
        results, skipped = _canned_results()
        frag_ids = ["conv-001_s000_x_f000", "conv-001_s000_x_f001"]

        with (
            patch("jarvis.cli.Fragmenter") as MockFragmenter,
            patch("jarvis.cli._build_memory_layer") as mock_build,
        ):
            MockFragmenter.return_value.fragment_conversation_extracts.return_value = (
                results, skipped
            )
            mock_memory = MagicMock()
            mock_memory.store.get_fragment.return_value = None
            mock_memory.persist_fragment_with_links.side_effect = frag_ids
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        assert mock_memory.persist_fragment_with_links.call_count == len(results)
        assert mock_memory.index_fragment_in_qdrant.call_count == len(results)

        # Ordering: index_in_qdrant must receive the fragment_id from persist
        _, out0 = results[0]
        _, out1 = results[1]
        mock_memory.index_fragment_in_qdrant.assert_any_call(
            fragment_id=frag_ids[0], output_data=out0
        )
        mock_memory.index_fragment_in_qdrant.assert_any_call(
            fragment_id=frag_ids[1], output_data=out1
        )

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
            mock_memory.store.get_fragment.return_value = {
                "fragment_id": "stub",  # existing row; CLI uses its derived fragment_id
                "qdrant_point_id": None,
            }
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        mock_memory.persist_fragment_with_links.assert_not_called()
        # Both fragments should be indexed (they have no qdrant_point_id)
        assert mock_memory.index_fragment_in_qdrant.call_count == len(results)
        _, out0 = results[0]
        _, out1 = results[1]
        # CLI derives fragment_id deterministically from segment_id + fragment_index
        mock_memory.index_fragment_in_qdrant.assert_any_call(
            fragment_id="conv-001_s000_x_f000", output_data=out0
        )
        mock_memory.index_fragment_in_qdrant.assert_any_call(
            fragment_id="conv-001_s000_x_f001", output_data=out1
        )

    def test_already_fully_indexed_skipped(self, tmp_path):
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
            mock_memory.store.get_fragment.return_value = {
                "fragment_id": "conv-001_s000_x_f000",
                "qdrant_point_id": "abc-123",
            }
            mock_build.return_value = mock_memory

            rc = cmd_fragment_extracts(args, config)

        assert rc == 0
        mock_memory.persist_fragment_with_links.assert_not_called()
        mock_memory.index_fragment_in_qdrant.assert_not_called()


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
