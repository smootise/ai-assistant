"""Tests for the web UI Flask routes."""

import pytest

from jarvis.store import SummaryStore
from jarvis.web.app import create_app
from tests.fixtures.web_seed import seed


@pytest.fixture
def seeded_store(tmp_path):
    s = SummaryStore(db_path=str(tmp_path / "test.db"))
    ids = seed(s)
    return s, ids


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = SummaryStore(db_path=db_path)
    ids = seed(s)

    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
    })
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, ids


def test_dashboard_200(client):
    c, _ = client
    resp = c.get("/")
    assert resp.status_code == 200
    data = resp.data.decode()
    assert "Dashboard" in data
    assert "Source Files" in data


def test_dashboard_shows_counts(client):
    c, _ = client
    resp = c.get("/")
    data = resp.data.decode()
    # 1 source, 1 conversation, 2 segments etc — values rendered in stat cards
    assert ">1<" in data or "1</div>" in data or "1\n" in data


def test_sources_list_200(client):
    c, _ = client
    resp = c.get("/sources")
    assert resp.status_code == 200
    assert b"test_export.json" in resp.data


def test_source_detail_200(client):
    c, ids = client
    resp = c.get(f"/sources/{ids['source_id']}")
    assert resp.status_code == 200
    assert b"test_export.json" in resp.data
    assert b"Test Conversation" in resp.data


def test_source_detail_404(client):
    c, _ = client
    resp = c.get("/sources/does-not-exist")
    assert resp.status_code == 404


def test_conversations_list_200(client):
    c, _ = client
    resp = c.get("/conversations")
    assert resp.status_code == 200
    assert b"Test Conversation" in resp.data


def test_conversation_detail_200_with_segments(client):
    c, ids = client
    resp = c.get(f"/conversations/{ids['conv_id']}")
    assert resp.status_code == 200
    data = resp.data.decode()
    assert "Test Conversation" in data
    assert "Segment" in data


def test_conversation_detail_segments_in_order(client):
    c, ids = client
    resp = c.get(f"/conversations/{ids['conv_id']}")
    data = resp.data.decode()
    pos0 = data.find(">0<")
    pos1 = data.find(">1<")
    assert pos0 < pos1, "Segment #0 should appear before Segment #1"


def test_conversation_detail_404(client):
    c, _ = client
    resp = c.get("/conversations/no-such-conv")
    assert resp.status_code == 404


def test_segment_detail_200(client):
    c, ids = client
    resp = c.get(f"/segments/{ids['seg0_id']}")
    assert resp.status_code == 200
    assert b"Hello" in resp.data


def test_segment_detail_shows_extract_link(client):
    c, ids = client
    resp = c.get(f"/segments/{ids['seg0_id']}")
    assert resp.status_code == 200
    assert b"extract_detail" in resp.data or b"/extracts/" in resp.data


def test_segment_detail_no_extract_renders_placeholder(client):
    c, ids = client
    # seg1 has no extract
    resp = c.get(f"/segments/{ids['seg1_id']}")
    assert resp.status_code == 200
    assert b"No extract yet" in resp.data


def test_segment_detail_404(client):
    c, _ = client
    resp = c.get("/segments/no-such-seg")
    assert resp.status_code == 404


def test_extract_detail_200(client):
    c, ids = client
    resp = c.get(f"/extracts/{ids['extract_id']}")
    assert resp.status_code == 200
    data = resp.data.decode()
    assert "Hello" in data
    assert "Hi there!" in data


def test_extract_detail_statements_ordered(client):
    c, ids = client
    resp = c.get(f"/extracts/{ids['extract_id']}")
    data = resp.data.decode()
    pos_hello = data.find("Hello")
    pos_hi = data.find("Hi there!")
    assert pos_hello < pos_hi, "Statement 0 should precede statement 1"


def test_extract_detail_shows_fragments(client):
    c, ids = client
    resp = c.get(f"/extracts/{ids['extract_id']}")
    data = resp.data.decode()
    assert "Greeting exchange" in data


def test_extract_detail_404(client):
    c, _ = client
    resp = c.get("/extracts/no-such-extract")
    assert resp.status_code == 404


def test_fragment_detail_200(client):
    c, ids = client
    resp = c.get(f"/fragments/{ids['frag0_id']}")
    assert resp.status_code == 200
    data = resp.data.decode()
    assert "Greeting exchange" in data
    assert "Hello" in data


def test_fragment_detail_shows_linked_statements(client):
    c, ids = client
    resp = c.get(f"/fragments/{ids['frag0_id']}")
    data = resp.data.decode()
    assert "Hi there!" in data
    assert "How are you?" not in data  # belongs to frag1, not frag0


def test_fragment_detail_404(client):
    c, _ = client
    resp = c.get("/fragments/no-such-frag")
    assert resp.status_code == 404


def test_empty_sources_list(tmp_path):
    db_path = str(tmp_path / "empty.db")
    SummaryStore(db_path=db_path)
    app = create_app({"db_path": db_path, "repo_root": str(tmp_path), "output_root": "OUTPUTS"})
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/sources")
        assert resp.status_code == 200
        assert b"No source files" in resp.data


def test_empty_conversations_list(tmp_path):
    db_path = str(tmp_path / "empty.db")
    SummaryStore(db_path=db_path)
    app = create_app({"db_path": db_path, "repo_root": str(tmp_path), "output_root": "OUTPUTS"})
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/conversations")
        assert resp.status_code == 200
        assert b"No conversations" in resp.data


def test_empty_dashboard(tmp_path):
    db_path = str(tmp_path / "empty.db")
    SummaryStore(db_path=db_path)
    app = create_app({"db_path": db_path, "repo_root": str(tmp_path), "output_root": "OUTPUTS"})
    app.config["TESTING"] = True
    with app.test_client() as c:
        resp = c.get("/")
        assert resp.status_code == 200
        assert b"No data ingested" in resp.data
