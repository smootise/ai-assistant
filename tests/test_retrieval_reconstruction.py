"""Tests for full fragment reconstruction from SQLite.

Seeds the relational chain (source_file → conversation → segment → extract
→ statements → fragment → links) and asserts that get_fragments_with_statements
returns all expected data.
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from jarvis.store import SummaryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> SummaryStore:
    return SummaryStore(db_path=str(tmp_path / "test.db"))


def _seed_full_chain(store: SummaryStore, conv_id: str = "conv-001") -> Dict[str, Any]:
    """Seed one complete chain and return IDs."""
    sha = "a" * 64
    store.insert_source_file({
        "source_file_id": sha,
        "source_kind": "chatgpt_raw_export",
        "original_filename": "export.json",
        "storage_path": "inbox/export.json",
        "sha256": sha,
        "size_bytes": 512,
    })
    store.insert_conversation({
        "conversation_id": conv_id,
        "source_platform": "chatgpt",
        "raw_source_file_id": sha,
        "title": "Test conv",
        "message_count": 2,
    })
    seg_id = f"{conv_id}_s000"
    store.insert_segment({
        "segment_id": seg_id,
        "conversation_id": conv_id,
        "segment_index": 0,
        "start_position": 0,
        "end_position": 1,
        "message_ids": ["msg-a", "msg-b"],
        "segment_text": "user: hello\n\nassistant: hi there",
    })
    extract_id = f"{seg_id}_x"
    store.insert_extract({
        "segment_id": seg_id,
        "segment_index": 0,
        "parent_conversation_id": conv_id,
        "provider": "local",
        "model": "gemma4:31b",
        "status": "ok",
    })
    statements = [
        {"statement_index": 0, "speaker": "user", "text": "hello"},
        {"statement_index": 1, "speaker": "assistant", "text": "hi there"},
    ]
    store.insert_statements(extract_id, statements)

    frag_id = f"{extract_id}_f000"
    store.insert_fragment({
        "extract_id": extract_id,
        "fragment_index": 0,
        "title": "Greeting",
        "retrieval_text": "user: hello\n\nassistant: hi there",
        "status": "ok",
    })
    statement_ids = [f"{extract_id}_st0000", f"{extract_id}_st0001"]
    store.insert_fragment_links(frag_id, statement_ids)

    return {
        "conv_id": conv_id,
        "seg_id": seg_id,
        "extract_id": extract_id,
        "frag_id": frag_id,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFragmentReconstruction:
    def test_full_chain_reconstructed(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _seed_full_chain(store)

        rows = store.get_fragments_with_statements([ids["frag_id"]])
        assert len(rows) == 1
        row = rows[0]

        # Fragment metadata
        assert row["fragment_id"] == ids["frag_id"]
        assert row["title"] == "Greeting"

        # Relational chain fields
        assert row["segment_id"] == ids["seg_id"]
        assert row["parent_conversation_id"] == "conv-001"
        assert row["segment_text"] == "user: hello\n\nassistant: hi there"

        # Statements reconstructed in link order
        stmts = row["statements"]
        assert len(stmts) == 2
        assert stmts[0]["speaker"] == "user"
        assert stmts[0]["text"] == "hello"
        assert stmts[0]["position_in_fragment"] == 0
        assert stmts[1]["speaker"] == "assistant"
        assert stmts[1]["text"] == "hi there"
        assert stmts[1]["position_in_fragment"] == 1

    def test_multiple_fragments_preserved_in_request_order(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _seed_full_chain(store)

        extract_id = ids["extract_id"]
        frag_id_1 = f"{extract_id}_f001"
        store.insert_fragment({
            "extract_id": extract_id,
            "fragment_index": 1,
            "title": "Second",
            "retrieval_text": "user: hello again",
            "status": "ok",
        })
        store.insert_fragment_links(frag_id_1, [f"{extract_id}_st0000"])

        # Request in reverse order
        rows = store.get_fragments_with_statements([frag_id_1, ids["frag_id"]])
        assert rows[0]["fragment_id"] == frag_id_1
        assert rows[1]["fragment_id"] == ids["frag_id"]

    def test_fragment_with_qdrant_point_id(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _seed_full_chain(store)
        store.update_fragment_embedding(ids["frag_id"], "pt-uuid-001", "qwen3-embedding")
        rows = store.get_fragments_with_statements([ids["frag_id"]])
        assert rows[0]["qdrant_point_id"] == "pt-uuid-001"

    def test_missing_fragment_id_silently_omitted(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _seed_full_chain(store)
        rows = store.get_fragments_with_statements(["nonexistent-id", ids["frag_id"]])
        # Only the real fragment is returned
        assert len(rows) == 1
        assert rows[0]["fragment_id"] == ids["frag_id"]

    def test_empty_input_returns_empty(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.get_fragments_with_statements([]) == []

    def test_cascade_delete_removes_links(self, tmp_path):
        store = _make_store(tmp_path)
        ids = _seed_full_chain(store)
        store.delete_fragments("conv-001")
        with store._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM fragment_statement_links"
            ).fetchone()[0]
        assert count == 0

    def test_schema_version_is_eight(self, tmp_path):
        store = _make_store(tmp_path)
        with store._connect() as conn:
            row = conn.execute(
                "SELECT value FROM _jarvis_meta WHERE key = 'schema_version'"
            ).fetchone()
        assert row is not None
        assert row[0] == "8"
