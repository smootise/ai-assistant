"""Tests for the ChunkSummarizer.

All Ollama calls are mocked — no live inference required.
"""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from jarvis.chunk_summarizer import ChunkSummarizer
from jarvis.store import SummaryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_JSON_RESPONSE = json.dumps({
    "summary": "The team decided to use qwen3-embedding for multilingual support.",
    "bullets": ["Decision: Use qwen3-embedding over nomic-embed-text for multilingual support."],
    "action_items": ["Pull qwen3-embedding model via Ollama"],
    "confidence": 0.85,
})


def _make_chunk(index: int, conversation_id: str = "test-conv") -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "chunk_id": f"{conversation_id}_c{index:03d}",
        "chunk_index": index,
        "start_position": index * 3,
        "end_position": index * 3 + 2,
        "message_ids": [f"id_{index}_0", f"id_{index}_1", f"id_{index}_2"],
        "chunk_text": f"user: Question {index}\n\nassistant: Answer {index}",
    }


def _make_summarizer(prompts_dir: str, context_window: int = 3) -> ChunkSummarizer:
    client = MagicMock()
    client.model = "test-model"
    client.generate.return_value = (_VALID_JSON_RESPONSE, False, "")
    client.parse_json_response.return_value = (json.loads(_VALID_JSON_RESPONSE), False, "")
    return ChunkSummarizer(
        ollama_client=client,
        prompts_dir=prompts_dir,
        schema="jarvis.summarization",
        schema_version="1.0.0",
        context_window=context_window,
    )


@pytest.fixture()
def prompts_dir(tmp_path) -> str:
    p = tmp_path / "prompts"
    p.mkdir()
    (p / "summarize_ai_chat_chunk.md").write_text(
        "You are JARVIS. Summarize the chunk.", encoding="utf-8"
    )
    return str(p)


@pytest.fixture()
def summarizer(prompts_dir) -> ChunkSummarizer:
    return _make_summarizer(prompts_dir)


# ---------------------------------------------------------------------------
# _build_chunk_prompt
# ---------------------------------------------------------------------------


class TestBuildChunkPrompt:
    def test_no_context_omits_context_block(self, summarizer):
        prompt = summarizer._build_chunk_prompt("user: Q\n\nassistant: A", [])
        assert "---BEGIN PREVIOUS CONTEXT---" not in prompt
        assert "---END PREVIOUS CONTEXT---" not in prompt

    def test_no_context_includes_transcript_block(self, summarizer):
        prompt = summarizer._build_chunk_prompt("user: Q\n\nassistant: A", [])
        assert "---BEGIN CHUNK TRANSCRIPT---" in prompt
        assert "user: Q" in prompt

    def test_with_context_includes_both_blocks(self, summarizer):
        prompt = summarizer._build_chunk_prompt("user: Q", ["Summary of chunk 0."])
        assert "---BEGIN PREVIOUS CONTEXT---" in prompt
        assert "---END PREVIOUS CONTEXT---" in prompt
        assert "---BEGIN CHUNK TRANSCRIPT---" in prompt

    def test_context_summaries_are_prefixed(self, summarizer):
        prompt = summarizer._build_chunk_prompt("user: Q", ["First.", "Second."])
        assert "Chunk 0: First." in prompt
        assert "Chunk 1: Second." in prompt

    def test_context_window_respected(self, prompts_dir):
        s = _make_summarizer(prompts_dir, context_window=2)
        summaries = ["S0", "S1", "S2", "S3", "S4"]
        # ChunkSummarizer slices via caller passing prior[-context_window:]
        # But _build_chunk_prompt itself accepts whatever is passed in
        prompt = s._build_chunk_prompt("user: Q", summaries[-2:])
        assert "Chunk 0: S3" in prompt
        assert "Chunk 1: S4" in prompt
        assert "S0" not in prompt
        assert "S1" not in prompt


# ---------------------------------------------------------------------------
# summarize_chunk
# ---------------------------------------------------------------------------


class TestSummarizeChunk:
    def test_happy_path_output_fields(self, summarizer, tmp_path):
        chunk = _make_chunk(0)
        output_dir, output_data = summarizer.summarize_chunk(
            chunk=chunk,
            prior_summaries=[],
            chunk_summaries_dir=tmp_path / "chunk_summaries",
        )
        assert output_data["chunk_id"] == chunk["chunk_id"]
        assert output_data["chunk_index"] == 0
        assert output_data["parent_conversation_id"] == "test-conv"
        assert output_data["source_kind"] == "ai_chat_chunk"
        assert output_data["status"] == "ok"

    def test_degraded_json_sets_status(self, prompts_dir, tmp_path):
        client = MagicMock()
        client.model = "test-model"
        fenced = f"```json\n{_VALID_JSON_RESPONSE}\n```"
        client.generate.return_value = (fenced, False, "")
        client.parse_json_response.return_value = (
            json.loads(_VALID_JSON_RESPONSE), True, "Stripped code fences"
        )
        s = ChunkSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        _, output_data = s.summarize_chunk(
            chunk=_make_chunk(0),
            prior_summaries=[],
            chunk_summaries_dir=tmp_path / "chunk_summaries",
        )
        assert output_data["status"] == "degraded"
        assert len(output_data.get("warnings", [])) > 0

    def test_output_files_written(self, summarizer, tmp_path):
        chunk = _make_chunk(5)
        chunk_summaries_dir = tmp_path / "chunk_summaries"
        summarizer.summarize_chunk(
            chunk=chunk,
            prior_summaries=[],
            chunk_summaries_dir=chunk_summaries_dir,
        )
        assert (chunk_summaries_dir / f"{chunk['chunk_id']}.json").exists()
        assert (chunk_summaries_dir / f"{chunk['chunk_id']}.md").exists()


# ---------------------------------------------------------------------------
# summarize_conversation_chunks
# ---------------------------------------------------------------------------


class TestSummarizeConversationChunks:
    def _write_chunks(self, chunks_dir: Path, chunks):
        chunks_dir.mkdir(parents=True)
        for c in chunks:
            (chunks_dir / f"chunk_{c['chunk_index']:03d}.json").write_text(
                json.dumps(c), encoding="utf-8"
            )

    def test_rolling_context_passed_correctly(self, prompts_dir, tmp_path):
        """Prompt for chunk 2 must contain the summary from chunk 0."""
        chunks = [_make_chunk(i) for i in range(3)]
        chunks_dir = tmp_path / "chunks"
        self._write_chunks(chunks_dir, chunks)

        client = MagicMock()
        client.model = "test-model"
        responses = [
            (json.dumps({
                "summary": f"Summary of chunk {i}.",
                "bullets": [], "action_items": [], "confidence": 0.8
            }), False, "")
            for i in range(3)
        ]
        client.generate.side_effect = responses
        client.parse_json_response.side_effect = [
            (json.loads(r[0]), False, "") for r in responses
        ]

        s = ChunkSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
            context_window=3,
        )
        s.summarize_conversation_chunks(
            chunks_dir=chunks_dir,
            conversation_id="test-conv",
            output_root=tmp_path / "OUTPUTS",
        )

        # Third generate call (chunk 2) must have summaries of chunk 0 and chunk 1 in prompt
        third_call_prompt = client.generate.call_args_list[2][0][0]
        assert "Summary of chunk 0." in third_call_prompt
        assert "Summary of chunk 1." in third_call_prompt

    def test_skips_pending_tail(self, prompts_dir, tmp_path):
        chunks = [_make_chunk(0)]
        chunks_dir = tmp_path / "chunks"
        self._write_chunks(chunks_dir, chunks)
        (chunks_dir / "pending_tail.json").write_text("{}", encoding="utf-8")

        client = MagicMock()
        client.model = "test-model"
        client.generate.return_value = (_VALID_JSON_RESPONSE, False, "")
        client.parse_json_response.return_value = (json.loads(_VALID_JSON_RESPONSE), False, "")

        s = ChunkSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        results = s.summarize_conversation_chunks(
            chunks_dir=chunks_dir,
            conversation_id="test-conv",
            output_root=tmp_path / "OUTPUTS",
        )
        assert len(results) == 1  # only the real chunk, not pending_tail
        assert client.generate.call_count == 1

    def test_raises_if_chunks_dir_missing(self, summarizer, tmp_path):
        with pytest.raises(FileNotFoundError):
            summarizer.summarize_conversation_chunks(
                chunks_dir=tmp_path / "nonexistent",
                conversation_id="x",
                output_root=tmp_path / "OUTPUTS",
            )

    def test_from_to_chunk_range(self, prompts_dir, tmp_path):
        chunks = [_make_chunk(i) for i in range(5)]
        chunks_dir = tmp_path / "chunks"
        self._write_chunks(chunks_dir, chunks)

        client = MagicMock()
        client.model = "test-model"
        client.generate.return_value = (_VALID_JSON_RESPONSE, False, "")
        client.parse_json_response.return_value = (json.loads(_VALID_JSON_RESPONSE), False, "")

        s = ChunkSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        results = s.summarize_conversation_chunks(
            chunks_dir=chunks_dir,
            conversation_id="test-conv",
            output_root=tmp_path / "OUTPUTS",
            from_chunk=1,
            to_chunk=3,
        )
        assert len(results) == 3
        assert client.generate.call_count == 3

    def test_preseed_context_from_existing_files(self, prompts_dir, tmp_path):
        """from_chunk=2 with existing summaries for 0 and 1 pre-seeds context."""
        chunks = [_make_chunk(i) for i in range(3)]
        chunks_dir = tmp_path / "chunks"
        self._write_chunks(chunks_dir, chunks)

        # Write existing summaries for chunks 0 and 1
        conv_id = "test-conv"
        summaries_dir = tmp_path / "OUTPUTS" / conv_id / "chunk_summaries"
        summaries_dir.mkdir(parents=True)
        for i in range(2):
            data = {
                "summary": f"Pre-existing summary {i}.",
                "bullets": [], "action_items": [], "confidence": 0.8,
                "source_file": f"{conv_id}_c{i:03d}.json",
                "source_kind": "ai_chat_chunk",
                "status": "ok",
                "provider": "local", "model": "m",
                "schema": "jarvis.summarization", "schema_version": "1.0.0",
                "created_at": "2026-01-01T00:00:00Z",
            }
            (summaries_dir / f"{conv_id}_c{i:03d}.json").write_text(
                json.dumps(data), encoding="utf-8"
            )

        client = MagicMock()
        client.model = "test-model"
        client.generate.return_value = (_VALID_JSON_RESPONSE, False, "")
        client.parse_json_response.return_value = (json.loads(_VALID_JSON_RESPONSE), False, "")

        s = ChunkSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
            context_window=3,
        )
        s.summarize_conversation_chunks(
            chunks_dir=chunks_dir,
            conversation_id=conv_id,
            output_root=tmp_path / "OUTPUTS",
            from_chunk=2,
        )

        prompt_used = client.generate.call_args_list[0][0][0]
        assert "Pre-existing summary 0." in prompt_used
        assert "Pre-existing summary 1." in prompt_used


# ---------------------------------------------------------------------------
# SummaryStore migration and chunk columns
# ---------------------------------------------------------------------------


class TestSummaryStoreChunkColumns:
    def _minimal_output(self, **extra) -> Dict[str, Any]:
        data = {
            "summary": "Test summary",
            "bullets": ["Bullet 1"],
            "action_items": [],
            "confidence": 0.8,
            "provider": "local",
            "model": "test",
            "schema": "jarvis.summarization",
            "schema_version": "1.0.0",
            "status": "ok",
            "source_file": "test.json",
            "source_kind": "conversation",
            "created_at": "2026-01-01T00:00:00Z",
        }
        data.update(extra)
        return data

    def test_chunk_columns_stored_and_retrieved(self, tmp_path):
        store = SummaryStore(str(tmp_path / "test.db"))
        output_data = self._minimal_output(
            source_kind="ai_chat_chunk",
            chunk_id="conv123_c005",
            chunk_index=5,
            parent_conversation_id="conv123",
        )
        row_id = store.insert_summary(output_data)
        rows = store.get_by_ids([row_id])
        assert rows[0]["chunk_id"] == "conv123_c005"
        assert rows[0]["chunk_index"] == 5
        assert rows[0]["parent_conversation_id"] == "conv123"

    def test_get_chunk_summaries_by_conversation(self, tmp_path):
        store = SummaryStore(str(tmp_path / "test.db"))
        # Insert 3 chunk summaries + 1 conversation summary
        for i in range(3):
            store.insert_summary(self._minimal_output(
                source_kind="ai_chat_chunk",
                chunk_id=f"conv_c{i:03d}",
                chunk_index=i,
                parent_conversation_id="conv",
            ))
        store.insert_summary(self._minimal_output(source_kind="conversation"))

        results = store.get_chunk_summaries_by_conversation("conv")
        assert len(results) == 3
        assert [r["chunk_index"] for r in results] == [0, 1, 2]

    def test_conversation_summary_not_in_chunk_query(self, tmp_path):
        store = SummaryStore(str(tmp_path / "test.db"))
        store.insert_summary(self._minimal_output(source_kind="conversation"))
        results = store.get_chunk_summaries_by_conversation("conv")
        assert results == []

    def test_migration_idempotent_on_fresh_db(self, tmp_path):
        """Opening a fresh store twice must not raise errors."""
        db_path = str(tmp_path / "test.db")
        SummaryStore(db_path)
        SummaryStore(db_path)  # second open triggers migration check again
