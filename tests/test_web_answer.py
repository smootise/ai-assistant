"""Tests for the /answer web route."""

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


def _make_app(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = SummaryStore(db_path=db_path)
    seed(s)
    prompt_file = tmp_path / "answer_question.md"
    prompt_file.write_text(
        "## Question\n{question}\n## Context\n{context_block}\n{user_context}\n## Answer\n",
        encoding="utf-8",
    )
    app = create_app({
        "db_path": db_path,
        "repo_root": str(tmp_path),
        "output_root": "OUTPUTS",
        "embedding_model": "qwen3-embedding",
        "ollama_base_url": "http://localhost:11434",
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "local_model_name": "gemma4:31b",
        "ollama_timeout": 60,
        "prompts_dir": str(tmp_path),
        "user_name": "",
    })
    app.config["TESTING"] = True
    return app


def test_answer_get_empty_form(tmp_path):
    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.get("/answer")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Answer a Question" in body
    assert "citations-list" not in body


@patch("jarvis.web.services.OllamaClient")
@patch("jarvis.web.services.run_retrieval")
@patch("jarvis.web.services.build_memory_layer")
def test_answer_happy_path(mock_bml, mock_retrieval, mock_ollama_cls, tmp_path):
    mock_bml.return_value = MagicMock()
    mock_retrieval.return_value = (FAKE_ROWS, {})
    mock_ollama = MagicMock()
    mock_ollama.chat.return_value = ("SQLite is the source of truth.", False, "")
    mock_ollama_cls.return_value = mock_ollama

    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.post("/answer", data={"query": "Why SQLite?", "top_k": "10", "temperature": "0.3"})

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "SQLite is the source of truth." in body
    assert "/fragments/conv_001_s000_x_f000" in body
    assert "/fragments/conv_002_s000_x_f000" in body
    assert "Storage layer discussion" in body


@patch("jarvis.web.services.run_retrieval")
@patch("jarvis.web.services.build_memory_layer")
def test_answer_empty_hits(mock_bml, mock_retrieval, tmp_path):
    mock_bml.return_value = MagicMock()
    mock_retrieval.return_value = ([], {})

    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.post("/answer", data={"query": "nonsense", "top_k": "10", "temperature": "0.3"})

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "No fragments matched" in body


@patch("jarvis.web.services.OllamaClient")
@patch("jarvis.web.services.run_retrieval")
@patch("jarvis.web.services.build_memory_layer")
def test_answer_degraded_response(mock_bml, mock_retrieval, mock_ollama_cls, tmp_path):
    mock_bml.return_value = MagicMock()
    mock_retrieval.return_value = (FAKE_ROWS, {})
    mock_ollama = MagicMock()
    mock_ollama.chat.return_value = ("Partial answer.", True, "model degraded")
    mock_ollama_cls.return_value = mock_ollama

    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.post("/answer", data={"query": "test", "top_k": "10", "temperature": "0.3"})

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "model degraded" in body
    assert "alert-warning" in body
    assert "Partial answer." in body


@patch("jarvis.web.services.OllamaClient")
@patch("jarvis.web.services.run_retrieval")
@patch("jarvis.web.services.build_memory_layer")
def test_answer_ollama_error(mock_bml, mock_retrieval, mock_ollama_cls, tmp_path):
    mock_bml.return_value = MagicMock()
    mock_retrieval.return_value = (FAKE_ROWS, {})
    mock_ollama = MagicMock()
    mock_ollama.chat.side_effect = RuntimeError("model not available")
    mock_ollama_cls.return_value = mock_ollama

    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.post("/answer", data={"query": "test", "top_k": "10", "temperature": "0.3"})

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "model not available" in body
    assert "alert-error" in body


@patch("jarvis.web.services.build_memory_layer")
def test_answer_qdrant_error(mock_bml, tmp_path):
    mock_memory = MagicMock()
    mock_memory.embedder.embed.return_value = [0.1] * 128
    mock_memory.vector_store.search.side_effect = RuntimeError("qdrant unavailable")
    mock_bml.return_value = mock_memory

    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.post("/answer", data={"query": "test", "top_k": "10", "temperature": "0.3"})

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "qdrant unavailable" in body
    assert "alert-error" in body


def test_answer_nav_link_present(tmp_path):
    app = _make_app(tmp_path)
    with app.test_client() as c:
        resp = c.get("/answer")
    assert b"/answer" in resp.data
    assert b"Answer" in resp.data
