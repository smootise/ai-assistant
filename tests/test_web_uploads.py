"""Tests for the upload routes."""

import io
from unittest.mock import patch

import pytest

from jarvis.store import SummaryStore
from jarvis.web.app import create_app


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test.db")
    SummaryStore(db_path=db_path)
    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
    })
    app.config["TESTING"] = True
    app.config["JARVIS_INBOX_RAW_DIR"] = tmp_path / "raw"
    with app.test_client() as c:
        yield c, tmp_path, db_path


def _valid_json_bytes():
    return b'{"conversation_id": "c1", "title": "T", "mapping": {}, "current_node": null}'


def test_upload_form_get(client):
    c, _, _ = client
    resp = c.get("/upload")
    assert resp.status_code == 200
    assert b"Upload" in resp.data
    assert b"chatgpt" in resp.data


def test_upload_no_file(client):
    c, _, _ = client
    resp = c.post("/upload", data={"source_type": "chatgpt"}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"No file selected" in resp.data


def test_upload_invalid_extension(client, tmp_path):
    c, raw_dir, _ = client
    data = {"source_type": "chatgpt", "file": (io.BytesIO(b"hello"), "export.txt")}
    resp = c.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b".txt" in resp.data or b"Only .json" in resp.data
    # No file should have been saved
    assert not any((tmp_path / "raw").iterdir()) if (tmp_path / "raw").exists() else True


def test_upload_unsupported_source_type(client):
    c, _, _ = client
    data = {
        "source_type": "conversation",
        "file": (io.BytesIO(_valid_json_bytes()), "export.json"),
    }
    resp = c.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"Unsupported source type" in resp.data or b"supported" in resp.data.lower()


def test_upload_valid_file_creates_job(client):
    c, tmp_path, db_path = client
    json_bytes = _valid_json_bytes()

    with patch("jarvis.web.routes.uploads.threading.Thread") as MockThread:
        mock_thread = MockThread.return_value
        data = {
            "source_type": "chatgpt",
            "file": (io.BytesIO(json_bytes), "my_export.json"),
        }
        resp = c.post("/upload", data=data, content_type="multipart/form-data")

    assert resp.status_code == 303
    location = resp.headers.get("Location", "")
    assert "/jobs/" in location

    # Verify file was saved under raw dir
    raw_dir = tmp_path / "raw"
    saved = list(raw_dir.iterdir())
    assert len(saved) == 1
    assert saved[0].suffix == ".json"
    assert "my_export" in saved[0].name

    # Verify job row exists
    store = SummaryStore(db_path=db_path)
    job_id = location.split("/jobs/")[-1]
    job = store.get_job(job_id)
    assert job is not None
    assert job["status"] == "pending"
    assert job["job_type"] == "ingest_chatgpt"
    assert job["input_metadata"]["original_filename"] == "my_export.json"

    # Thread was started
    mock_thread.start.assert_called_once()
