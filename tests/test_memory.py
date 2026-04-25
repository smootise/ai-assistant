"""Unit tests for the memory layer — relational store and memory orchestration.

Tests do not require live Ollama or Qdrant. External calls are mocked.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from jarvis.embedder import build_embedding_text
from jarvis.memory import MemoryLayer
from jarvis.store import SummaryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_store(tmp_path: Path) -> SummaryStore:
    return SummaryStore(db_path=str(tmp_path / "test.db"))


def _source_file_data(kind: str = "chatgpt_raw_export") -> Dict[str, Any]:
    sha = "abc123def456" * 3 + "abcd"  # 40 hex chars
    return {
        "source_file_id": sha,
        "source_kind": kind,
        "original_filename": "conv.json",
        "storage_path": "inbox/ai_chat/chatgpt/conv-001/conv.json",
        "sha256": sha,
        "size_bytes": 1024,
    }


def _conversation_data(conv_id: str = "conv-001") -> Dict[str, Any]:
    return {
        "conversation_id": conv_id,
        "source_platform": "chatgpt",
        "title": "Test conversation",
        "message_count": 4,
    }


def _segment_data(conv_id: str = "conv-001", idx: int = 0) -> Dict[str, Any]:
    seg_id = f"{conv_id}_s{idx:03d}"
    return {
        "segment_id": seg_id,
        "conversation_id": conv_id,
        "segment_index": idx,
        "start_position": idx * 3,
        "end_position": idx * 3 + 2,
        "message_ids": [f"msg-{idx}-a", f"msg-{idx}-b"],
        "segment_text": f"user: hello {idx}\n\nassistant: hi there {idx}",
    }


def _extract_data(conv_id: str = "conv-001", seg_idx: int = 0) -> Dict[str, Any]:
    seg_id = f"{conv_id}_s{seg_idx:03d}"
    return {
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
                "text": "hi there",
            },
        ],
    }


def _fragment_data(conv_id: str = "conv-001", seg_idx: int = 0, frag_idx: int = 0) -> Dict[str, Any]:
    seg_id = f"{conv_id}_s{seg_idx:03d}"
    extract_id = f"{seg_id}_x"
    return {
        # Fields used by store.insert_fragment
        "extract_id": extract_id,
        "fragment_index": frag_idx,
        "title": f"Fragment {frag_idx}",
        "retrieval_text": "user: hello\n\nassistant: hi there",
        # Fields used by memory.index_fragment_in_qdrant (embedding text + payload)
        "text": "user: hello\n\nassistant: hi there",
        "segment_id": seg_id,
        "segment_index": seg_idx,
        "parent_conversation_id": conv_id,
        "conversation_date": "2026-01-01T00:00:00Z",
        "statements": [
            {"statement_index": 0, "speaker": "user", "text": "hello"},
            {"statement_index": 1, "speaker": "assistant", "text": "hi there"},
        ],
        "status": "ok",
        "provider": "local",
        "model": "gemma4:31b",
    }


# ---------------------------------------------------------------------------
# SummaryStore tests
# ---------------------------------------------------------------------------


class TestSummaryStore:
    def test_insert_source_file(self, tmp_path):
        store = _make_store(tmp_path)
        data = _source_file_data()
        fid = store.insert_source_file(data)
        assert fid == data["source_file_id"]

    def test_insert_source_file_idempotent(self, tmp_path):
        store = _make_store(tmp_path)
        data = _source_file_data()
        store.insert_source_file(data)
        store.insert_source_file(data)  # should not raise
        with store._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
        assert count == 1

    def test_insert_conversation(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        with store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = 'conv-001'"
            ).fetchone()
        assert row is not None

    def test_insert_segment(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        seg = _segment_data()
        seg_id = store.insert_segment(seg)
        assert seg_id == seg["segment_id"]

    def test_get_segment(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        seg = _segment_data()
        store.insert_segment(seg)
        fetched = store.get_segment(seg["segment_id"])
        assert fetched is not None
        assert fetched["segment_text"] == seg["segment_text"]
        assert fetched["message_ids"] == seg["message_ids"]

    def test_insert_segment_idempotent(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        seg = _segment_data()
        store.insert_segment(seg)
        store.insert_segment(seg)
        with store._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        assert count == 1

    def test_insert_extract_and_statements(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        store.insert_segment(_segment_data())
        ext = _extract_data()
        extract_id = store.insert_extract(ext)
        assert extract_id == "conv-001_s000_x"
        store.insert_statements(extract_id, ext["statements"])
        stmts = store.get_statements_for_extract(extract_id)
        assert len(stmts) == 2
        assert stmts[0]["speaker"] == "user"
        assert stmts[1]["speaker"] == "assistant"

    def test_insert_extract_idempotent(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        store.insert_segment(_segment_data())
        ext = _extract_data()
        store.insert_extract(ext)
        store.insert_extract(ext)
        with store._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM extracts").fetchone()[0]
        assert count == 1

    def test_insert_fragment_and_links(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        store.insert_segment(_segment_data())
        ext = _extract_data()
        extract_id = store.insert_extract(ext)
        store.insert_statements(extract_id, ext["statements"])
        frag = _fragment_data()
        fragment_id = store.insert_fragment(frag)
        assert fragment_id == "conv-001_s000_x_f000"
        statement_ids = [f"{extract_id}_st0000", f"{extract_id}_st0001"]
        store.insert_fragment_links(fragment_id, statement_ids)
        with store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM fragment_statement_links WHERE fragment_id = ?",
                (fragment_id,),
            ).fetchone()[0]
        assert count == 2

    def test_update_fragment_embedding(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        store.insert_segment(_segment_data())
        extract_id = store.insert_extract(_extract_data())
        frag = _fragment_data()
        fragment_id = store.insert_fragment(frag)
        store.update_fragment_embedding(fragment_id, "qdrant-uuid-001", "qwen3-embedding")
        fetched = store.get_fragment(fragment_id)
        assert fetched["qdrant_point_id"] == "qdrant-uuid-001"
        assert fetched["embedding_model"] == "qwen3-embedding"
        assert fetched["embedded_at"] is not None

    def test_get_fragments_with_statements_reconstruction(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        store.insert_segment(_segment_data())
        ext = _extract_data()
        extract_id = store.insert_extract(ext)
        store.insert_statements(extract_id, ext["statements"])
        frag = _fragment_data()
        fragment_id = store.insert_fragment(frag)
        statement_ids = [f"{extract_id}_st0000", f"{extract_id}_st0001"]
        store.insert_fragment_links(fragment_id, statement_ids)

        rows = store.get_fragments_with_statements([fragment_id])
        assert len(rows) == 1
        row = rows[0]
        assert row["fragment_id"] == fragment_id
        assert row["segment_id"] == "conv-001_s000"
        assert row["segment_text"] is not None
        assert len(row["statements"]) == 2
        assert row["statements"][0]["speaker"] == "user"
        assert row["statements"][1]["speaker"] == "assistant"

    def test_get_fragments_with_statements_preserves_order(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        store.insert_segment(_segment_data())
        ext = _extract_data()
        extract_id = store.insert_extract(ext)
        store.insert_statements(extract_id, ext["statements"])

        frag0 = _fragment_data(frag_idx=0)
        frag1 = {**frag0, "fragment_index": 1, "title": "Second"}
        fid0 = store.insert_fragment(frag0)
        fid1 = store.insert_fragment(frag1)

        rows = store.get_fragments_with_statements([fid1, fid0])
        assert rows[0]["fragment_id"] == fid1
        assert rows[1]["fragment_id"] == fid0

    def test_get_fragments_with_statements_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_fragments_with_statements([]) == []

    def test_delete_fragments(self, tmp_path):
        store = _make_store(tmp_path)
        store.insert_source_file(_source_file_data())
        store.insert_conversation(_conversation_data())
        store.insert_segment(_segment_data())
        ext = _extract_data()
        extract_id = store.insert_extract(ext)
        store.insert_statements(extract_id, ext["statements"])
        frag = _fragment_data()
        fragment_id = store.insert_fragment(frag)
        store.update_fragment_embedding(fragment_id, "pt-uuid", "qwen3-embedding")

        point_ids = store.delete_fragments("conv-001")
        assert "pt-uuid" in point_ids
        with store._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# build_embedding_text (kept for coverage — function is still importable)
# ---------------------------------------------------------------------------


class TestBuildEmbeddingText:
    def test_fragment_returns_text_field(self):
        data = {
            "source_kind": "ai_chat_fragment",
            "text": "user: hello\n\nassistant: hi",
        }
        assert build_embedding_text(data) == "user: hello\n\nassistant: hi"

    def test_non_fragment_combines_fields(self):
        data = {
            "source_kind": "conversation",
            "summary": "A summary.",
            "bullets": ["Point A"],
            "action_items": ["Do X"],
        }
        text = build_embedding_text(data)
        assert "A summary." in text
        assert "Point A" in text
        assert "Do X" in text
        assert text.index("Key points") < text.index("A summary.")

    def test_empty_arrays_handled(self):
        data = {
            "source_kind": "conversation",
            "summary": "Only summary.",
            "bullets": [],
            "action_items": [],
        }
        text = build_embedding_text(data)
        assert "Only summary." in text
        assert "Key points" not in text


# ---------------------------------------------------------------------------
# MemoryLayer tests
# ---------------------------------------------------------------------------


class TestMemoryLayer:
    def _make_memory(self, tmp_path: Path) -> MemoryLayer:
        store = _make_store(tmp_path)
        mock_vs = MagicMock()
        mock_vs.upsert.return_value = "fake-point-uuid"
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 1024
        mock_embedder.model = "qwen3-embedding"
        return MemoryLayer(store=store, vector_store=mock_vs, embedder=mock_embedder)

    def _seed_extract(self, memory: MemoryLayer) -> str:
        """Insert source_file, conversation, segment, and extract; return extract_id."""
        memory.persist_source_file(_source_file_data())
        memory.persist_conversation(_conversation_data())
        memory.persist_segment(_segment_data())
        extract_id = memory.persist_extract_with_statements(_extract_data())
        return extract_id

    def test_persist_extract_with_statements(self, tmp_path):
        memory = self._make_memory(tmp_path)
        extract_id = self._seed_extract(memory)
        assert extract_id == "conv-001_s000_x"
        stmts = memory.store.get_statements_for_extract(extract_id)
        assert len(stmts) == 2

    def test_persist_fragment_with_links(self, tmp_path):
        memory = self._make_memory(tmp_path)
        extract_id = self._seed_extract(memory)
        frag_data = _fragment_data()
        frag_id = memory.persist_fragment_with_links(frag_data, extract_id=extract_id)
        assert frag_id == "conv-001_s000_x_f000"
        with memory.store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM fragment_statement_links WHERE fragment_id = ?",
                (frag_id,),
            ).fetchone()[0]
        assert count == 2

    def test_index_fragment_in_qdrant(self, tmp_path):
        memory = self._make_memory(tmp_path)
        extract_id = self._seed_extract(memory)
        frag_data = _fragment_data()
        frag_id = memory.persist_fragment_with_links(frag_data, extract_id=extract_id)
        point_id = memory.index_fragment_in_qdrant(fragment_id=frag_id, output_data=frag_data)
        assert point_id == "fake-point-uuid"
        # Qdrant upsert called with fragment_id in payload
        call_kwargs = memory.vector_store.upsert.call_args.kwargs
        assert call_kwargs["fragment_id"] == frag_id
        # Written back to SQLite
        fetched = memory.store.get_fragment(frag_id)
        assert fetched["qdrant_point_id"] == "fake-point-uuid"

    def test_index_fragment_embedding_failure_raises(self, tmp_path):
        memory = self._make_memory(tmp_path)
        extract_id = self._seed_extract(memory)
        frag_id = memory.persist_fragment_with_links(_fragment_data(), extract_id=extract_id)
        memory.embedder.embed.side_effect = RuntimeError("Ollama down")
        with pytest.raises(RuntimeError, match="Ollama down"):
            memory.index_fragment_in_qdrant(frag_id, _fragment_data())

    def test_index_fragment_qdrant_failure_raises(self, tmp_path):
        memory = self._make_memory(tmp_path)
        extract_id = self._seed_extract(memory)
        frag_id = memory.persist_fragment_with_links(_fragment_data(), extract_id=extract_id)
        memory.vector_store.upsert.side_effect = RuntimeError("Qdrant unavailable")
        with pytest.raises(RuntimeError, match="Qdrant unavailable"):
            memory.index_fragment_in_qdrant(frag_id, _fragment_data())

    def test_persist_idempotent_on_rerun(self, tmp_path):
        """Running persist twice on the same data produces exactly one row each."""
        memory = self._make_memory(tmp_path)
        extract_id = self._seed_extract(memory)
        frag_data = _fragment_data()
        memory.persist_fragment_with_links(frag_data, extract_id=extract_id)
        memory.persist_fragment_with_links(frag_data, extract_id=extract_id)
        with memory.store._connect() as conn:
            frag_count = conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
            link_count = conn.execute(
                "SELECT COUNT(*) FROM fragment_statement_links"
            ).fetchone()[0]
        assert frag_count == 1
        assert link_count == 2
