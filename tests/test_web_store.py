"""Tests for the read-only SummaryStore methods added for the web layer."""

import pytest

from jarvis.store import SummaryStore
from tests.fixtures.web_seed import seed


@pytest.fixture
def store(tmp_path):
    s = SummaryStore(db_path=str(tmp_path / "test.db"))
    seed(s)
    return s


def test_count_all_returns_expected_counts(store):
    counts = store.count_all()
    assert counts["source_files"] == 1
    assert counts["conversations"] == 1
    assert counts["segments"] == 2
    assert counts["extracts"] == 1
    assert counts["extract_statements"] == 3
    assert counts["fragments"] == 2


def test_count_all_empty_db(tmp_path):
    empty = SummaryStore(db_path=str(tmp_path / "empty.db"))
    counts = empty.count_all()
    assert all(v == 0 for v in counts.values())


def test_list_source_files(store):
    rows = store.list_source_files()
    assert len(rows) == 1
    assert rows[0]["original_filename"] == "test_export.json"


def test_get_source_file_found(store):
    src = store.get_source_file("src_test_001")
    assert src is not None
    assert src["source_kind"] == "chatgpt"


def test_get_source_file_missing(store):
    assert store.get_source_file("does-not-exist") is None


def test_list_conversations(store):
    rows = store.list_conversations()
    assert len(rows) == 1
    assert rows[0]["title"] == "Test Conversation"


def test_get_conversation_found(store):
    conv = store.get_conversation("conv_test_001")
    assert conv is not None
    assert conv["source_platform"] == "chatgpt"


def test_get_conversation_missing(store):
    assert store.get_conversation("no-such-conv") is None


def test_list_conversations_for_source(store):
    rows = store.list_conversations_for_source("src_test_001")
    assert len(rows) == 1
    assert rows[0]["conversation_id"] == "conv_test_001"


def test_list_conversations_for_source_empty(store):
    assert store.list_conversations_for_source("unknown-src") == []


def test_list_segments_for_conversation_ordered(store):
    segs = store.list_segments_for_conversation("conv_test_001")
    assert len(segs) == 2
    assert segs[0]["segment_index"] == 0
    assert segs[1]["segment_index"] == 1


def test_list_segments_for_conversation_empty(store):
    assert store.list_segments_for_conversation("no-conv") == []


def test_get_extract_found(store, tmp_path):
    segs = store.list_segments_for_conversation("conv_test_001")
    extract = store.get_extract_by_segment(segs[0]["segment_id"])
    assert extract is not None
    via_id = store.get_extract(extract["extract_id"])
    assert via_id is not None
    assert via_id["extract_id"] == extract["extract_id"]


def test_get_extract_missing(store):
    assert store.get_extract("no-such-extract") is None


def test_list_fragments_for_extract(store):
    segs = store.list_segments_for_conversation("conv_test_001")
    extract = store.get_extract_by_segment(segs[0]["segment_id"])
    frags = store.list_fragments_for_extract(extract["extract_id"])
    assert len(frags) == 2
    assert frags[0]["fragment_index"] == 0
    assert frags[1]["fragment_index"] == 1


def test_list_fragments_for_extract_empty(store):
    assert store.list_fragments_for_extract("no-extract") == []


def test_recent_records_structure(store):
    recent = store.recent_records(5)
    assert "source_files" in recent
    assert "conversations" in recent
    assert "segments" in recent
    assert "extracts" in recent
    assert "fragments" in recent
    assert len(recent["source_files"]) == 1
    assert recent["source_files"][0]["id"] == "src_test_001"


def test_recent_records_empty_db(tmp_path):
    empty = SummaryStore(db_path=str(tmp_path / "empty.db"))
    recent = empty.recent_records(5)
    assert all(v == [] for v in recent.values())
