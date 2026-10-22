"""Cleanup service failures must reach command/job handlers rather than report success."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis import cli
from jarvis.web.extract_runner import _wipe_extracts_for_force
from jarvis.web.fragment_runner import _wipe_fragments_for_force


@pytest.mark.parametrize("stage", ["extract", "fragment"])
def test_web_cleanup_propagates_qdrant_failure(tmp_path: Path, stage: str) -> None:
    config = {"qdrant_collection": "jarvis_mom_fragments"}
    store = MagicMock()
    store.delete_extracts.return_value = ["point"]
    store.delete_fragments.return_value = ["point"]
    vectors = MagicMock()
    vectors.delete_points.side_effect = RuntimeError("Qdrant delete failed")
    with patch(f"jarvis.web.{stage}_runner.create_vector_store", return_value=vectors) as create:
        with pytest.raises(RuntimeError, match="Qdrant delete failed"):
            if stage == "extract":
                _wipe_extracts_for_force(store, tmp_path, tmp_path, "conv", 0, 0, True, config)
            else:
                _wipe_fragments_for_force(store, tmp_path, "conv", config)
    create.assert_called_once_with(config)
    vectors.delete_points.assert_called_once_with(["point"])


@pytest.mark.parametrize("stage", ["extract", "fragment"])
def test_cli_cleanup_failure_stops_processing(tmp_path: Path, stage: str) -> None:
    config = {"output_root": str(tmp_path / "output"), "qdrant_collection": "jarvis_mom_fragments"}
    (tmp_path / "inbox" / "conv" / "segments").mkdir(parents=True)
    (tmp_path / "output" / "conv" / "extracts").mkdir(parents=True)
    args = argparse.Namespace(
        source="chatgpt",
        conversation_id="conv",
        force=True,
        persist=True,
        embed=True,
        from_segment=0,
        to_segment=0,
        inbox_dir=str(tmp_path / "inbox"),
    )
    store = MagicMock()
    store.delete_extracts.return_value = ["point"]
    store.delete_fragments.return_value = ["point"]
    vectors = MagicMock()
    vectors.delete_points.side_effect = RuntimeError("Qdrant delete failed")
    with (
        patch("jarvis.cli.create_summary_store", return_value=store),
        patch("jarvis.cli.create_vector_store", return_value=vectors) as create,
        patch("jarvis.cli.create_ollama_client") as inference,
    ):
        handler = cli.cmd_extract_segments if stage == "extract" else cli.cmd_fragment_extracts
        assert handler(args, config) == 1
    create.assert_called_once_with(config)
    inference.assert_not_called()
