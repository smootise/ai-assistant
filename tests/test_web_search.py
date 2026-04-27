"""Tests for the /search web route."""

from unittest.mock import MagicMock, patch

import pytest

from jarvis.store import SummaryStore
from jarvis.web.app import create_app
from tests.fixtures.web_seed import seed


FAKE_ROWS = [
    {
        "fragment_id": "conv_001_s000_x_f000",
        "extract_id": "conv_001_s000_x",
        "segment_id": "conv_001_s000",
        "parent_conversation_id": "conv_001",
        "fragment_index": 0,
        "title": "Storage layer discussion",
        "retrieval_text": "We discussed the storage layer.",
        "statements": [{"speaker": "assistant", "text": "Decision: Use SQLite as source of truth"}],
        "conversation_date": "2025-01-01",
        "created_at": "2025-01-01T00:00:00Z",
        "status": "ok",
        "qdrant_point_id": None,
        "embedded_at": None,
        "embedding_model": None,
    },
    {
        "fragment_id": "conv_002_s000_x_f000",
        "extract_id": "conv_002_s000_x",
        "segment_id": "conv_002_s000",
        "parent_conversation_id": "conv_002",
        "fragment_index": 0,
        "title": "Vector store comparison",
        "retrieval_text": "We compared Qdrant vs Pinecone.",
        "statements": [{"speaker": "assistant", "text": "Decision: Use Qdrant for local-first"}],
        "conversation_date": "2025-01-02",
        "created_at": "2025-01-02T00:00:00Z",
        "status": "ok",
        "qdrant_point_id": None,
        "embedded_at": None,
        "embedding_model": None,
    },
]

FAKE_SCORES = {
    "conv_001_s000_x_f000": 0.92,
    "conv_002_s000_x_f000": 0.81,
}


def _make_app(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = SummaryStore(db_path=db_path)
    seed(s)
    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
        "embedding_model": "qwen3-embedding",
        "ollama_base_url": "http://localhost:11434",
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
    })
    app.config["TESTING"] = True
    return app


def test_search_empty_form(tmp_path):
    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.get("/search")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Search Fragments" in body
    assert "result-item" not in body


@patch("jarvis.web.services.run_retrieval")
@patch("jarvis.web.services.build_memory_layer")
def test_search_shows_results(mock_bml, mock_retrieval, tmp_path):
    mock_bml.return_value = MagicMock()
    mock_retrieval.return_value = (FAKE_ROWS, FAKE_SCORES)

    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.get("/search?query=storage+layer&top_k=10")

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Storage layer discussion" in body
    assert "0.9200" in body
    assert "Vector store comparison" in body
    assert "0.8100" in body
    assert "/fragments/conv_001_s000_x_f000" in body
    assert "/extracts/conv_001_s000_x" in body
    assert "/segments/conv_001_s000" in body


@patch("jarvis.web.services.run_retrieval")
@patch("jarvis.web.services.build_memory_layer")
def test_search_empty_results(mock_bml, mock_retrieval, tmp_path):
    mock_bml.return_value = MagicMock()
    mock_retrieval.return_value = ([], {})

    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.get("/search?query=nonsense")

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "No fragments matched" in body
    assert "result-item" not in body


@patch("jarvis.web.services.build_memory_layer")
def test_search_qdrant_error(mock_bml, tmp_path):
    mock_memory = MagicMock()
    mock_memory.embedder.embed.return_value = [0.1] * 128
    mock_memory.vector_store.search.side_effect = RuntimeError("qdrant down")
    mock_bml.return_value = mock_memory

    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.get("/search?query=foo")

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "qdrant down" in body
    assert "alert-error" in body


def test_search_nav_link_present(tmp_path):
    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.get("/search")
    assert b"/search" in resp.data
    assert b"Search" in resp.data
