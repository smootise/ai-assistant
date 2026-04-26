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
    assert counts["jobs"] == 0


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


# ------------------------------------------------------------------
# Job tracking
# ------------------------------------------------------------------

@pytest.fixture
def job_store(tmp_path):
    return SummaryStore(db_path=str(tmp_path / "jobs.db"))


def test_create_job_returns_id(job_store):
    job_id = job_store.create_job("ingest_chatgpt", {"file": "a.json"})
    assert job_id.startswith("job_")
    assert len(job_id) > 4


def test_get_job_pending(job_store):
    job_id = job_store.create_job("ingest_chatgpt", {"file": "a.json"})
    job = job_store.get_job(job_id)
    assert job is not None
    assert job["status"] == "pending"
    assert job["job_type"] == "ingest_chatgpt"
    assert job["input_metadata"]["file"] == "a.json"
    assert job["result"] is None
    assert job["error"] is None
    assert job["started_at"] is None
    assert job["finished_at"] is None


def test_mark_job_running(job_store):
    job_id = job_store.create_job("ingest_chatgpt", {})
    job_store.mark_job_running(job_id)
    job = job_store.get_job(job_id)
    assert job["status"] == "running"
    assert job["started_at"] is not None


def test_mark_job_succeeded(job_store):
    job_id = job_store.create_job("ingest_chatgpt", {})
    job_store.mark_job_running(job_id)
    job_store.mark_job_succeeded(job_id, {"conversation_id": "c1", "segment_count": 2})
    job = job_store.get_job(job_id)
    assert job["status"] == "succeeded"
    assert job["result"]["conversation_id"] == "c1"
    assert job["finished_at"] is not None


def test_mark_job_failed(job_store):
    job_id = job_store.create_job("ingest_chatgpt", {})
    job_store.mark_job_running(job_id)
    job_store.mark_job_failed(job_id, "Traceback: ValueError")
    job = job_store.get_job(job_id)
    assert job["status"] == "failed"
    assert "ValueError" in job["error"]
    assert job["finished_at"] is not None


def test_get_job_not_found(job_store):
    assert job_store.get_job("nonexistent") is None


def test_list_jobs_empty(job_store):
    assert job_store.list_jobs() == []


def test_list_jobs_order(job_store):
    j1 = job_store.create_job("ingest_chatgpt", {"n": 1})
    j2 = job_store.create_job("ingest_chatgpt", {"n": 2})
    jobs = job_store.list_jobs()
    ids = [j["job_id"] for j in jobs]
    assert j1 in ids and j2 in ids
    # newest first — j2 was created after j1
    assert ids.index(j2) < ids.index(j1)


def test_count_jobs(job_store):
    assert job_store.count_jobs() == 0
    job_store.create_job("ingest_chatgpt", {})
    job_store.create_job("ingest_chatgpt", {})
    assert job_store.count_jobs() == 2
