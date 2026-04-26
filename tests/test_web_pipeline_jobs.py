"""Tests for the extract + fragment pipeline job routes."""

from unittest.mock import patch

import pytest

from jarvis.store import SummaryStore
from jarvis.web.app import create_app


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = SummaryStore(db_path=db_path)
    _seed_conversation(store, "conv1")
    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
    })
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, db_path, store


@pytest.fixture
def client_with_extracts(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = SummaryStore(db_path=db_path)
    _seed_conversation(store, "conv1")
    _seed_extract(store, "conv1", segment_index=0)
    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
    })
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, db_path, store


def _seed_conversation(store: SummaryStore, conversation_id: str) -> None:
    with store._connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO conversations
               (conversation_id, title, source_platform, created_at)
               VALUES (?, 'Test Conv', 'chatgpt', '2024-01-01T00:00:00Z')""",
            (conversation_id,),
        )


def _seed_extract(store: SummaryStore, conversation_id: str, segment_index: int = 0) -> str:
    segment_id = f"{conversation_id}_seg{segment_index:03d}"
    with store._connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO segments
               (segment_id, conversation_id, segment_index,
                start_position, end_position, message_ids_json, segment_text, created_at)
               VALUES (?, ?, ?, 0, 0, '[]', '', '2024-01-01T00:00:00Z')""",
            (segment_id, conversation_id, segment_index),
        )
    extract_id = f"{segment_id}_x"
    with store._connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO extracts
               (extract_id, segment_id, segment_index, parent_conversation_id,
                provider, model, status, created_at)
               VALUES (?, ?, ?, ?, 'local', 'test', 'ok', '2024-01-01T00:00:00Z')""",
            (extract_id, segment_id, segment_index, conversation_id),
        )
    return extract_id


# ---------------------------------------------------------------------------
# Extract form GET
# ---------------------------------------------------------------------------

def test_extract_form_get(client):
    c, _, _ = client
    resp = c.get("/conversations/conv1/extract")
    assert resp.status_code == 200
    assert b"Run Extraction" in resp.data
    assert b"conv1" in resp.data


def test_extract_form_get_missing_conversation(client):
    c, _, _ = client
    resp = c.get("/conversations/no-such-conv/extract")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Extract submit POST
# ---------------------------------------------------------------------------

def test_extract_submit_valid_creates_job(client):
    c, db_path, _ = client
    with patch("jarvis.web.routes.pipeline_jobs.threading.Thread") as MockThread:
        mock_thread = MockThread.return_value
        resp = c.post(
            "/conversations/conv1/extract",
            data={"persist": "on"},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 303
    location = resp.headers.get("Location", "")
    assert "/jobs/" in location

    store = SummaryStore(db_path=db_path)
    job_id = location.split("/jobs/")[-1]
    job = store.get_job(job_id)
    assert job is not None
    assert job["status"] == "pending"
    assert job["job_type"] == "extract_segments"
    assert job["input_metadata"]["conversation_id"] == "conv1"
    assert job["input_metadata"]["persist"] is True
    mock_thread.start.assert_called_once()


def test_extract_submit_with_range(client):
    c, db_path, _ = client
    with patch("jarvis.web.routes.pipeline_jobs.threading.Thread"):
        resp = c.post(
            "/conversations/conv1/extract",
            data={"from_segment": "2", "to_segment": "5", "persist": "on"},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 303
    store = SummaryStore(db_path=db_path)
    location = resp.headers.get("Location", "")
    job = store.get_job(location.split("/jobs/")[-1])
    assert job["input_metadata"]["from_segment"] == 2
    assert job["input_metadata"]["to_segment"] == 5


def test_extract_submit_invalid_range(client):
    c, _, _ = client
    resp = c.post(
        "/conversations/conv1/extract",
        data={"from_segment": "5", "to_segment": "2"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"from_segment" in resp.data or b"<=" in resp.data


def test_extract_submit_negative_segment(client):
    c, _, _ = client
    resp = c.post(
        "/conversations/conv1/extract",
        data={"from_segment": "-1"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_extract_submit_non_integer_segment(client):
    c, _, _ = client
    resp = c.post(
        "/conversations/conv1/extract",
        data={"from_segment": "abc"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_extract_submit_missing_conversation(client):
    c, _, _ = client
    resp = c.post(
        "/conversations/no-such/extract",
        data={},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Fragment form GET
# ---------------------------------------------------------------------------

def test_fragment_form_get_no_extracts_shows_warning(client):
    c, _, _ = client
    resp = c.get("/conversations/conv1/fragment")
    assert resp.status_code == 200
    assert b"No extracts found" in resp.data or b"extracts" in resp.data.lower()
    # Submit button must NOT be present
    assert b"Run fragmentation" not in resp.data


def test_fragment_form_get_with_extracts_shows_form(client_with_extracts):
    c, _, _ = client_with_extracts
    resp = c.get("/conversations/conv1/fragment")
    assert resp.status_code == 200
    assert b"Run fragmentation" in resp.data
    assert b"embed" in resp.data.lower()


def test_fragment_form_get_missing_conversation(client):
    c, _, _ = client
    resp = c.get("/conversations/no-such-conv/fragment")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Fragment submit POST
# ---------------------------------------------------------------------------

def test_fragment_submit_valid_creates_job(client_with_extracts):
    c, db_path, _ = client_with_extracts
    with patch("jarvis.web.routes.pipeline_jobs.threading.Thread") as MockThread:
        mock_thread = MockThread.return_value
        resp = c.post(
            "/conversations/conv1/fragment",
            data={"persist": "on", "embed": "on"},
            content_type="multipart/form-data",
        )

    assert resp.status_code == 303
    location = resp.headers.get("Location", "")
    assert "/jobs/" in location

    store = SummaryStore(db_path=db_path)
    job = store.get_job(location.split("/jobs/")[-1])
    assert job is not None
    assert job["status"] == "pending"
    assert job["job_type"] == "fragment_extracts"
    assert job["input_metadata"]["conversation_id"] == "conv1"
    assert job["input_metadata"]["persist"] is True
    assert job["input_metadata"]["embed"] is True
    mock_thread.start.assert_called_once()


def test_fragment_submit_no_extracts_rejected(client):
    c, _, _ = client
    resp = c.post(
        "/conversations/conv1/fragment",
        data={"persist": "on"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"extracts" in resp.data.lower()


def test_fragment_submit_embed_without_persist_rejected(client_with_extracts):
    c, _, _ = client_with_extracts
    resp = c.post(
        "/conversations/conv1/fragment",
        data={"embed": "on"},  # persist NOT checked
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert b"persist" in resp.data.lower()


def test_fragment_submit_invalid_range(client_with_extracts):
    c, _, _ = client_with_extracts
    resp = c.post(
        "/conversations/conv1/fragment",
        data={"from_segment": "10", "to_segment": "3", "persist": "on"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_fragment_submit_missing_conversation(client):
    c, _, _ = client
    resp = c.post(
        "/conversations/no-such/fragment",
        data={"persist": "on"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Job detail page — extract/fragment-specific rendering
# ---------------------------------------------------------------------------

def _make_app_with_jobs(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = SummaryStore(db_path=db_path)
    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
    })
    app.config["TESTING"] = True
    return app, store, db_path


def test_job_detail_extract_pending(tmp_path):
    app, store, _ = _make_app_with_jobs(tmp_path)
    job_id = store.create_job("extract_segments", {"conversation_id": "c1"})
    with app.test_client() as c:
        resp = c.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert b"extract_segments" in resp.data
    assert b"pending" in resp.data
    assert b'http-equiv="refresh"' in resp.data


def test_job_detail_extract_succeeded_shows_links(tmp_path):
    app, store, _ = _make_app_with_jobs(tmp_path)
    job_id = store.create_job("extract_segments", {
        "conversation_id": "c1",
        "from_segment": None,
        "to_segment": None,
        "force": False,
        "persist": True,
    })
    store.mark_job_running(job_id)
    store.mark_job_succeeded(job_id, {
        "conversation_id": "c1",
        "segments_processed": 3,
        "extracts_persisted": 2,
        "extract_ids": ["c1_seg000_x", "c1_seg001_x"],
        "extract_ids_truncated": False,
    })
    with app.test_client() as c:
        resp = c.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert b"succeeded" in resp.data
    assert b"c1_seg000_x" in resp.data
    assert b"c1_seg001_x" in resp.data
    assert b'http-equiv="refresh"' not in resp.data


def test_job_detail_fragment_succeeded_shows_links(tmp_path):
    app, store, _ = _make_app_with_jobs(tmp_path)
    job_id = store.create_job("fragment_extracts", {
        "conversation_id": "c1",
        "from_segment": None,
        "to_segment": None,
        "force": False,
        "persist": True,
        "embed": True,
    })
    store.mark_job_running(job_id)
    store.mark_job_succeeded(job_id, {
        "conversation_id": "c1",
        "fragments_produced": 4,
        "fragments_persisted": 4,
        "embedded": True,
        "skipped_segments": [],
        "fragment_ids": ["c1_seg000_x_f000", "c1_seg000_x_f001"],
        "fragment_ids_truncated": False,
    })
    with app.test_client() as c:
        resp = c.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert b"succeeded" in resp.data
    assert b"fragment" in resp.data.lower()
    assert b"c1_seg000_x_f000" in resp.data
    assert b'http-equiv="refresh"' not in resp.data


def test_job_detail_fragment_failed_shows_error(tmp_path):
    app, store, _ = _make_app_with_jobs(tmp_path)
    job_id = store.create_job("fragment_extracts", {"conversation_id": "c1"})
    store.mark_job_running(job_id)
    store.mark_job_failed(job_id, "FileNotFoundError: extracts dir not found")
    with app.test_client() as c:
        resp = c.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert b"failed" in resp.data
    assert b"FileNotFoundError" in resp.data
    assert b'http-equiv="refresh"' not in resp.data


def test_job_detail_fragment_skipped_segments(tmp_path):
    app, store, _ = _make_app_with_jobs(tmp_path)
    job_id = store.create_job("fragment_extracts", {"conversation_id": "c1"})
    store.mark_job_running(job_id)
    store.mark_job_succeeded(job_id, {
        "conversation_id": "c1",
        "fragments_produced": 0,
        "fragments_persisted": 0,
        "embedded": False,
        "skipped_segments": [["c1_seg000", "no statements"]],
        "fragment_ids": [],
        "fragment_ids_truncated": False,
    })
    with app.test_client() as c:
        resp = c.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert b"no statements" in resp.data
