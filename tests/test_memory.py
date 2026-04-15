"""Unit tests for the memory layer — store, embedder text builder, and memory orchestration.

These tests do not require live Ollama or Qdrant services. External calls are
replaced with simple stubs so the suite runs in CI without local services.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from jarvis.embedder import build_embedding_text
from jarvis.memory import MemoryLayer
from jarvis.store import SummaryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_output() -> Dict[str, Any]:
    """Load the expected_summary fixture as a dict."""
    with open(FIXTURE_DIR / "expected_summary.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def tmp_store(tmp_path: Path) -> SummaryStore:
    """SummaryStore backed by a temporary SQLite file."""
    return SummaryStore(db_path=str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# SummaryStore tests
# ---------------------------------------------------------------------------


class TestSummaryStore:
    def test_insert_returns_id(self, tmp_store: SummaryStore, sample_output: Dict[str, Any]):
        row_id = tmp_store.insert_summary(sample_output)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_inserted_row_is_retrievable(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        row_id = tmp_store.insert_summary(sample_output)
        rows = tmp_store.get_by_ids([row_id])
        assert len(rows) == 1
        row = rows[0]
        assert row["source_file"] == sample_output["source_file"]
        assert row["summary"] == sample_output["summary"]
        assert isinstance(row["bullets"], list)
        assert isinstance(row["action_items"], list)

    def test_artifact_paths_stored(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        row_id = tmp_store.insert_summary(
            sample_output,
            output_json_path="OUTPUTS/20260409/conv_tiny_test.json",
            output_md_path="OUTPUTS/20260409/conv_tiny_test.md",
        )
        rows = tmp_store.get_by_ids([row_id])
        assert rows[0]["output_json_path"] == "OUTPUTS/20260409/conv_tiny_test.json"
        assert rows[0]["output_md_path"] == "OUTPUTS/20260409/conv_tiny_test.md"

    def test_update_embedding(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        row_id = tmp_store.insert_summary(sample_output)
        tmp_store.update_embedding(
            summary_id=row_id,
            qdrant_point_id="test-uuid-1234",
            embedding_model="qwen3-embedding",
        )
        rows = tmp_store.get_by_ids([row_id])
        assert rows[0]["qdrant_point_id"] == "test-uuid-1234"
        assert rows[0]["embedding_model"] == "qwen3-embedding"
        assert rows[0]["embedded_at"] is not None

    def test_get_by_ids_preserves_order(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        id1 = tmp_store.insert_summary(sample_output)
        id2 = tmp_store.insert_summary({**sample_output, "source_file": "other.json"})
        # Request in reverse order
        rows = tmp_store.get_by_ids([id2, id1])
        assert rows[0]["id"] == id2
        assert rows[1]["id"] == id1

    def test_schema_fields_stored(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        row_id = tmp_store.insert_summary(sample_output)
        rows = tmp_store.get_by_ids([row_id])
        assert rows[0]["schema"] == "jarvis.summarization"
        assert rows[0]["schema_version"] == "1.0.0"

    def test_latency_ms_stored(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        row_id = tmp_store.insert_summary(sample_output)
        rows = tmp_store.get_by_ids([row_id])
        assert rows[0]["latency_ms"] == sample_output["latency_ms"]

    def test_get_by_ids_empty(self, tmp_store: SummaryStore):
        assert tmp_store.get_by_ids([]) == []


# ---------------------------------------------------------------------------
# build_embedding_text tests
# ---------------------------------------------------------------------------


class TestBuildEmbeddingText:
    def test_contains_all_semantic_fields(self, sample_output: Dict[str, Any]):
        text = build_embedding_text(sample_output)
        assert sample_output["summary"] in text
        assert sample_output["source_file"] in text
        assert sample_output["source_kind"] in text
        for bullet in sample_output["bullets"]:
            assert bullet in text
        for action in sample_output["action_items"]:
            assert action in text

    def test_no_raw_json_structure(self, sample_output: Dict[str, Any]):
        text = build_embedding_text(sample_output)
        assert "{" not in text
        assert '"summary"' not in text

    def test_preserves_original_language(self):
        """French text must not be translated."""
        fr_output = {
            "source_kind": "conversation",
            "source_file": "conv_fr.json",
            "summary": "L'équipe a finalisé les spécifications.",
            "bullets": ["Décision : deux sous-dossiers"],
            "action_items": ["Créer les fichiers FR"],
        }
        text = build_embedding_text(fr_output)
        assert "L'équipe" in text
        assert "Décision" in text

    def test_empty_arrays_handled(self):
        output = {
            "source_kind": "conversation",
            "source_file": "test.json",
            "summary": "A summary.",
            "bullets": [],
            "action_items": [],
        }
        text = build_embedding_text(output)
        assert "A summary." in text
        assert "Key points" not in text
        assert "Action items" not in text


# ---------------------------------------------------------------------------
# MemoryLayer tests (stubbed external services)
# ---------------------------------------------------------------------------


class TestMemoryLayer:
    def test_persist_full_flow(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        """persist() should insert into SQLite, call embed, upsert to Qdrant,
        and write back the Qdrant point ID."""
        fake_vector = [0.1] * 1024

        mock_vector_store = MagicMock()
        mock_vector_store.upsert.return_value = "fake-point-uuid"

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = fake_vector
        mock_embedder.model = "qwen3-embedding"

        memory = MemoryLayer(
            store=tmp_store,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        summary_id = memory.persist(output_data=sample_output, output_dir=None)

        # SQLite record exists
        rows = tmp_store.get_by_ids([summary_id])
        assert len(rows) == 1
        assert rows[0]["qdrant_point_id"] == "fake-point-uuid"
        assert rows[0]["embedding_model"] == "qwen3-embedding"
        assert rows[0]["embedded_at"] is not None

        # Qdrant upsert called once with the right summary_id
        mock_vector_store.upsert.assert_called_once()
        call_kwargs = mock_vector_store.upsert.call_args.kwargs
        assert call_kwargs["summary_id"] == summary_id
        assert call_kwargs["vector"] == fake_vector

    def test_persist_embedding_failure_raises(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        """If embedding fails, RuntimeError should propagate."""
        mock_vector_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = RuntimeError("Ollama down")
        mock_embedder.model = "qwen3-embedding"

        memory = MemoryLayer(
            store=tmp_store,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        with pytest.raises(RuntimeError, match="Ollama down"):
            memory.persist(output_data=sample_output)

    def test_persist_qdrant_failure_raises(
        self, tmp_store: SummaryStore, sample_output: Dict[str, Any]
    ):
        """If Qdrant upsert fails, RuntimeError should propagate."""
        mock_vector_store = MagicMock()
        mock_vector_store.upsert.side_effect = RuntimeError("Qdrant unavailable")

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.0] * 1024
        mock_embedder.model = "qwen3-embedding"

        memory = MemoryLayer(
            store=tmp_store,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

        with pytest.raises(RuntimeError, match="Qdrant unavailable"):
            memory.persist(output_data=sample_output)
