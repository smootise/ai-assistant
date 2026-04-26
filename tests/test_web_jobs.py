"""Tests for the jobs routes."""

import pytest

from jarvis.store import SummaryStore
from jarvis.web.app import create_app


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test.db")
    store = SummaryStore(db_path=db_path)

    job_pending = store.create_job("ingest_chatgpt", {"original_filename": "a.json"})
    store.mark_job_running(job_pending)

    job_ok = store.create_job("ingest_chatgpt", {"original_filename": "b.json"})
    store.mark_job_running(job_ok)
    store.mark_job_succeeded(job_ok, {
        "conversation_id": "conv_abc",
        "source_file_id": "src_abc",
        "segment_count": 3,
    })

    job_fail = store.create_job("ingest_chatgpt", {"original_filename": "c.json"})
    store.mark_job_running(job_fail)
    store.mark_job_failed(job_fail, "ValueError: parse error\n  line 2")

    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
    })
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, job_pending, job_ok, job_fail


def test_jobs_list_200(client):
    c, *_ = client
    resp = c.get("/jobs")
    assert resp.status_code == 200
    assert b"Jobs" in resp.data


def test_jobs_list_shows_all_jobs(client):
    c, job_pending, job_ok, job_fail, *_ = client
    resp = c.get("/jobs")
    data = resp.data.decode()
    assert "running" in data or "pending" in data
    assert "succeeded" in data
    assert "failed" in data


def test_jobs_list_empty(tmp_path):
    db_path = str(tmp_path / "empty.db")
    SummaryStore(db_path=db_path)
    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
    })
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/jobs")
        assert resp.status_code == 200
        assert b"No jobs yet" in resp.data


def test_job_detail_running_has_refresh(client):
    c, job_pending, *_ = client
    resp = c.get(f"/jobs/{job_pending}")
    assert resp.status_code == 200
    assert b'http-equiv="refresh"' in resp.data


def test_job_detail_succeeded_no_refresh(client):
    c, _, job_ok, *_ = client
    resp = c.get(f"/jobs/{job_ok}")
    assert resp.status_code == 200
    assert b'http-equiv="refresh"' not in resp.data
    data = resp.data.decode()
    assert "conv_abc" in data
    assert "succeeded" in data


def test_job_detail_failed_shows_error(client):
    c, _, _, job_fail = client
    resp = c.get(f"/jobs/{job_fail}")
    assert resp.status_code == 200
    assert b"parse error" in resp.data
    assert b"failed" in resp.data
    assert b'http-equiv="refresh"' not in resp.data


def test_job_detail_not_found(client):
    c, *_ = client
    resp = c.get("/jobs/job_doesnotexist")
    assert resp.status_code == 404
