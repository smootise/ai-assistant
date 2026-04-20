"""Tests for cmd_answer and _build_context_block."""

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.cli import _build_context_block, cmd_answer


FAKE_CONFIG = {
    "embedding_model": "qwen3-embedding",
    "ollama_base_url": "http://localhost:11434",
    "local_model_name": "gemma4:31b",
    "qdrant_host": "localhost",
    "qdrant_port": 6333,
    "db_path": "data/jarvis.db",
    "prompts_dir": "prompts",
    "ollama_timeout": 600,
}

FAKE_ROWS = [
    {
        "id": 1,
        "summary": "We discussed the storage layer.",
        "bullets": ["Decision: Use SQLite as source of truth"],
        "source_file": "conv_001.json",
        "created_at": "2025-01-01T00:00:00Z",
    },
    {
        "id": 2,
        "summary": "We compared Qdrant vs Pinecone.",
        "bullets": ["Decision: Use Qdrant for local-first"],
        "source_file": "conv_002.json",
        "created_at": "2025-01-02T00:00:00Z",
    },
]


def _make_args(query: str, top_k: int = 5, temperature: float = 0.3) -> argparse.Namespace:
    return argparse.Namespace(query=query, top_k=top_k, temperature=temperature)


def _mock_memory(hits, rows):
    memory = MagicMock()
    memory.embedder.embed.return_value = [0.1] * 128
    memory.vector_store.search.return_value = hits
    memory.store.get_by_ids.return_value = rows
    return memory


@patch("jarvis.cli.OllamaClient")
@patch("jarvis.cli._build_memory_layer")
def test_answer_happy_path(mock_build_memory, mock_ollama_cls, tmp_path, capsys):
    hits = [(1, 0.9, "pt1"), (2, 0.8, "pt2")]
    mock_build_memory.return_value = _mock_memory(hits, FAKE_ROWS)

    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = ("SQLite is the source of truth.", False, "")
    mock_ollama_cls.return_value = mock_ollama

    prompt_file = tmp_path / "answer_question.md"
    prompt_file.write_text(
        "## Question\n{question}\n## Context excerpts\n{context_block}\n## Answer\n",
        encoding="utf-8",
    )
    config = {**FAKE_CONFIG, "prompts_dir": str(tmp_path)}

    result = cmd_answer(_make_args("Why SQLite?"), config)

    assert result == 0
    out = capsys.readouterr().out
    assert "SQLite is the source of truth." in out
    assert 'Answer to: "Why SQLite?"' in out
    mock_ollama.generate.assert_called_once()


@patch("jarvis.cli.OllamaClient")
@patch("jarvis.cli._build_memory_layer")
def test_answer_no_hits(mock_build_memory, mock_ollama_cls, tmp_path, capsys):
    mock_build_memory.return_value = _mock_memory([], [])

    result = cmd_answer(_make_args("Unknown topic"), FAKE_CONFIG)

    assert result == 0
    out = capsys.readouterr().out
    assert "No relevant context found" in out
    mock_ollama_cls.return_value.generate.assert_not_called()


@patch("jarvis.cli.OllamaClient")
@patch("jarvis.cli._build_memory_layer")
def test_answer_degraded_response(mock_build_memory, mock_ollama_cls, tmp_path, caplog):
    hits = [(1, 0.9, "pt1")]
    mock_build_memory.return_value = _mock_memory(hits, FAKE_ROWS[:1])

    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = ("Some answer.", True, "JSON fallback used")
    mock_ollama_cls.return_value = mock_ollama

    prompt_file = tmp_path / "answer_question.md"
    prompt_file.write_text("{question}\n{context_block}", encoding="utf-8")
    config = {**FAKE_CONFIG, "prompts_dir": str(tmp_path)}

    with caplog.at_level(logging.WARNING):
        result = cmd_answer(_make_args("What?"), config)

    assert result == 0
    assert "Degraded response" in caplog.text


@patch("jarvis.cli.OllamaClient")
@patch("jarvis.cli._build_memory_layer")
def test_answer_connection_error(mock_build_memory, mock_ollama_cls):
    mock_build_memory.side_effect = ConnectionError("Qdrant unreachable")

    result = cmd_answer(_make_args("Any question"), FAKE_CONFIG)

    assert result == 1


def test_build_context_block_with_bullets():
    block = _build_context_block(FAKE_ROWS)
    assert "Excerpt 1" in block
    assert "source: conv_001.json" in block
    assert "Decision: Use SQLite as source of truth" in block
    assert "Excerpt 2" in block


def test_build_context_block_no_bullets():
    rows = [{"id": 1, "summary": "A summary.", "bullets": None,
             "source_file": "f.json", "created_at": "2025-01-01T00:00:00Z"}]
    block = _build_context_block(rows)
    assert "A summary." in block
    assert "Key points" not in block
