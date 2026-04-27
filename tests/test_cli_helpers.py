"""Tests for the extracted run_retrieval and generate_answer CLI helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.cli import run_retrieval, generate_answer


FAKE_ROWS = [
    {
        "fragment_id": "conv_001_s000_x_f000",
        "segment_id": "conv_001_s000",
        "title": "Storage layer discussion",
        "retrieval_text": "We discussed the storage layer.",
        "statements": [{"speaker": "assistant", "text": "Decision: Use SQLite as source of truth"}],
        "conversation_date": "2025-01-01",
        "fragment_created_at": "2025-01-01T00:00:00Z",
    },
]


def _make_memory(hits, rows):
    memory = MagicMock()
    memory.embedder.embed.return_value = [0.1] * 128
    memory.vector_store.search.return_value = hits
    memory.store.get_fragments_with_statements.return_value = rows
    return memory


def test_run_retrieval_returns_rows_and_scores():
    hits = [("conv_001_s000_x_f000", 0.9, "pt1")]
    memory = _make_memory(hits, FAKE_ROWS)

    rows, scores = run_retrieval(memory, "Why SQLite?", top_k=5, min_results=1, min_score=0.5)

    assert len(rows) == 1
    assert rows[0]["fragment_id"] == "conv_001_s000_x_f000"
    assert scores["conv_001_s000_x_f000"] == pytest.approx(0.9)


def test_run_retrieval_empty_hits():
    memory = _make_memory([], [])
    rows, scores = run_retrieval(memory, "nothing", top_k=5, min_results=1, min_score=0.5)
    assert rows == []
    assert scores == {}


def test_run_retrieval_hybrid_cutoff():
    hits = [
        ("frag_a", 0.9, "p1"),
        ("frag_b", 0.8, "p2"),
        ("frag_c", 0.3, "p3"),  # below min_score but within min_results
        ("frag_d", 0.1, "p4"),  # below min_score and beyond min_results=3
    ]
    rows_data = [{"fragment_id": f} for f in ["frag_a", "frag_b", "frag_c"]]
    memory = _make_memory(hits, rows_data)
    memory.store.get_fragments_with_statements.side_effect = lambda ids: [
        {"fragment_id": fid} for fid in ids
    ]

    rows, scores = run_retrieval(memory, "q", top_k=10, min_results=3, min_score=0.5)

    returned_ids = [r["fragment_id"] for r in rows]
    assert "frag_a" in returned_ids
    assert "frag_b" in returned_ids
    assert "frag_c" in returned_ids
    assert "frag_d" not in returned_ids


def test_generate_answer_returns_stripped_text(tmp_path):
    prompt_file = tmp_path / "answer_question.md"
    prompt_file.write_text(
        "Q: {question}\nCtx: {context_block}\n{user_context}\n## Answer\n",
        encoding="utf-8",
    )
    memory = MagicMock()
    ollama = MagicMock()
    ollama.chat.return_value = ("## Answer\nSQLite is the source of truth.", False, "")

    answer, is_degraded, warning = generate_answer(
        memory, ollama, "Why SQLite?", FAKE_ROWS, str(tmp_path), "", 0.3
    )

    assert answer == "SQLite is the source of truth."
    assert is_degraded is False
    assert warning == ""


def test_generate_answer_degraded(tmp_path):
    prompt_file = tmp_path / "answer_question.md"
    prompt_file.write_text("{question}{context_block}{user_context}", encoding="utf-8")
    memory = MagicMock()
    ollama = MagicMock()
    ollama.chat.return_value = ("Partial.", True, "timeout exceeded")

    answer, is_degraded, warning = generate_answer(
        memory, ollama, "q", FAKE_ROWS, str(tmp_path), "", 0.3
    )

    assert answer == "Partial."
    assert is_degraded is True
    assert "timeout" in warning


def test_generate_answer_user_context_injected(tmp_path):
    prompt_file = tmp_path / "answer_question.md"
    prompt_file.write_text("{user_context}", encoding="utf-8")
    memory = MagicMock()
    ollama = MagicMock()
    ollama.chat.return_value = ("ok", False, "")

    generate_answer(memory, ollama, "q", FAKE_ROWS, str(tmp_path), "Alice", 0.3)

    call_args = ollama.chat.call_args
    system_prompt = call_args[0][0]
    assert "Alice" in system_prompt
