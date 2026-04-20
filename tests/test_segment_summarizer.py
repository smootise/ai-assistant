"""Tests for the SegmentSummarizer.

All Ollama calls are mocked — no live inference required.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from jarvis.segment_summarizer import SegmentSummarizer
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


def _make_segment(index: int, conversation_id: str = "test-conv") -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "segment_id": f"{conversation_id}_s{index:03d}",
        "segment_index": index,
        "start_position": index * 3,
        "end_position": index * 3 + 2,
        "message_ids": [f"id_{index}_0", f"id_{index}_1", f"id_{index}_2"],
        "segment_text": f"user: Question {index}\n\nassistant: Answer {index}",
    }


def _make_summarizer(prompts_dir: str, context_window: int = 3) -> SegmentSummarizer:
    client = MagicMock()
    client.model = "test-model"
    client.generate.return_value = (_VALID_JSON_RESPONSE, False, "")
    client.parse_json_response.return_value = (json.loads(_VALID_JSON_RESPONSE), False, "")
    return SegmentSummarizer(
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
    (p / "summarize_ai_chat_segment.md").write_text(
        "You are JARVIS. Summarize the segment.", encoding="utf-8"
    )
    return str(p)


@pytest.fixture()
def summarizer(prompts_dir) -> SegmentSummarizer:
    return _make_summarizer(prompts_dir)


# ---------------------------------------------------------------------------
# _build_segment_prompt
# ---------------------------------------------------------------------------


class TestBuildSegmentPrompt:
    def test_no_context_omits_context_block(self, summarizer):
        prompt = summarizer._build_segment_prompt("user: Q\n\nassistant: A", [])
        assert "---BEGIN PREVIOUS CONTEXT---" not in prompt
        assert "---END PREVIOUS CONTEXT---" not in prompt

    def test_no_context_includes_transcript_block(self, summarizer):
        prompt = summarizer._build_segment_prompt("user: Q\n\nassistant: A", [])
        assert "---BEGIN SEGMENT TRANSCRIPT---" in prompt
        assert "user: Q" in prompt

    def test_with_context_includes_both_blocks(self, summarizer):
        prompt = summarizer._build_segment_prompt("user: Q", ["Summary of segment 0."])
        assert "---BEGIN PREVIOUS CONTEXT---" in prompt
        assert "---END PREVIOUS CONTEXT---" in prompt
        assert "---BEGIN SEGMENT TRANSCRIPT---" in prompt

    def test_context_summaries_are_prefixed(self, summarizer):
        prompt = summarizer._build_segment_prompt("user: Q", ["First.", "Second."])
        assert "Segment 0: First." in prompt
        assert "Segment 1: Second." in prompt

    def test_context_window_respected(self, prompts_dir):
        s = _make_summarizer(prompts_dir, context_window=2)
        summaries = ["S0", "S1", "S2", "S3", "S4"]
        prompt = s._build_segment_prompt("user: Q", summaries[-2:])
        assert "Segment 0: S3" in prompt
        assert "Segment 1: S4" in prompt
        assert "S0" not in prompt
        assert "S1" not in prompt


# ---------------------------------------------------------------------------
# summarize_segment
# ---------------------------------------------------------------------------


class TestSummarizeSegment:
    def test_happy_path_output_fields(self, summarizer, tmp_path):
        segment = _make_segment(0)
        output_dir, output_data = summarizer.summarize_segment(
            segment=segment,
            prior_summaries=[],
            segment_summaries_dir=tmp_path / "segment_summaries",
        )
        assert output_data["segment_id"] == segment["segment_id"]
        assert output_data["segment_index"] == 0
        assert output_data["parent_conversation_id"] == "test-conv"
        assert output_data["source_kind"] == "ai_chat_segment"
        assert output_data["status"] == "ok"

    def test_degraded_json_sets_status(self, prompts_dir, tmp_path):
        client = MagicMock()
        client.model = "test-model"
        fenced = f"```json\n{_VALID_JSON_RESPONSE}\n```"
        client.generate.return_value = (fenced, False, "")
        client.parse_json_response.return_value = (
            json.loads(_VALID_JSON_RESPONSE), True, "Stripped code fences"
        )
        s = SegmentSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        _, output_data = s.summarize_segment(
            segment=_make_segment(0),
            prior_summaries=[],
            segment_summaries_dir=tmp_path / "segment_summaries",
        )
        assert output_data["status"] == "degraded"
        assert len(output_data.get("warnings", [])) > 0

    def test_output_files_written(self, summarizer, tmp_path):
        segment = _make_segment(5)
        segment_summaries_dir = tmp_path / "segment_summaries"
        summarizer.summarize_segment(
            segment=segment,
            prior_summaries=[],
            segment_summaries_dir=segment_summaries_dir,
        )
        assert (segment_summaries_dir / f"{segment['segment_id']}.json").exists()
        assert (segment_summaries_dir / f"{segment['segment_id']}.md").exists()


# ---------------------------------------------------------------------------
# summarize_conversation_segments
# ---------------------------------------------------------------------------


class TestSummarizeConversationSegments:
    def _write_segments(self, segments_dir: Path, segments):
        segments_dir.mkdir(parents=True)
        for s in segments:
            (segments_dir / f"segment_{s['segment_index']:03d}.json").write_text(
                json.dumps(s), encoding="utf-8"
            )

    def test_rolling_context_passed_correctly(self, prompts_dir, tmp_path):
        segments = [_make_segment(i) for i in range(3)]
        segments_dir = tmp_path / "segments"
        self._write_segments(segments_dir, segments)

        client = MagicMock()
        client.model = "test-model"
        responses = [
            (json.dumps({
                "summary": f"Summary of segment {i}.",
                "bullets": [], "action_items": [], "confidence": 0.8
            }), False, "")
            for i in range(3)
        ]
        client.generate.side_effect = responses
        client.parse_json_response.side_effect = [
            (json.loads(r[0]), False, "") for r in responses
        ]

        s = SegmentSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
            context_window=3,
        )
        s.summarize_conversation_segments(
            segments_dir=segments_dir,
            conversation_id="test-conv",
            output_root=tmp_path / "OUTPUTS",
        )

        third_call_prompt = client.generate.call_args_list[2][0][0]
        assert "Summary of segment 0." in third_call_prompt
        assert "Summary of segment 1." in third_call_prompt

    def test_skips_pending_tail(self, prompts_dir, tmp_path):
        segments = [_make_segment(0)]
        segments_dir = tmp_path / "segments"
        self._write_segments(segments_dir, segments)
        (segments_dir / "pending_tail.json").write_text("{}", encoding="utf-8")

        client = MagicMock()
        client.model = "test-model"
        client.generate.return_value = (_VALID_JSON_RESPONSE, False, "")
        client.parse_json_response.return_value = (json.loads(_VALID_JSON_RESPONSE), False, "")

        s = SegmentSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        results = s.summarize_conversation_segments(
            segments_dir=segments_dir,
            conversation_id="test-conv",
            output_root=tmp_path / "OUTPUTS",
        )
        assert len(results) == 1
        assert client.generate.call_count == 1

    def test_raises_if_segments_dir_missing(self, summarizer, tmp_path):
        with pytest.raises(FileNotFoundError):
            summarizer.summarize_conversation_segments(
                segments_dir=tmp_path / "nonexistent",
                conversation_id="x",
                output_root=tmp_path / "OUTPUTS",
            )

    def test_from_to_segment_range(self, prompts_dir, tmp_path):
        segments = [_make_segment(i) for i in range(5)]
        segments_dir = tmp_path / "segments"
        self._write_segments(segments_dir, segments)

        client = MagicMock()
        client.model = "test-model"
        client.generate.return_value = (_VALID_JSON_RESPONSE, False, "")
        client.parse_json_response.return_value = (json.loads(_VALID_JSON_RESPONSE), False, "")

        s = SegmentSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        results = s.summarize_conversation_segments(
            segments_dir=segments_dir,
            conversation_id="test-conv",
            output_root=tmp_path / "OUTPUTS",
            from_segment=1,
            to_segment=3,
        )
        assert len(results) == 3
        assert client.generate.call_count == 3

    def test_preseed_context_from_existing_files(self, prompts_dir, tmp_path):
        segments = [_make_segment(i) for i in range(3)]
        segments_dir = tmp_path / "segments"
        self._write_segments(segments_dir, segments)

        conv_id = "test-conv"
        summaries_dir = tmp_path / "OUTPUTS" / conv_id / "segment_summaries"
        summaries_dir.mkdir(parents=True)
        for i in range(2):
            data = {
                "summary": f"Pre-existing summary {i}.",
                "bullets": [], "action_items": [], "confidence": 0.8,
                "source_file": f"{conv_id}_s{i:03d}.json",
                "source_kind": "ai_chat_segment",
                "status": "ok",
                "provider": "local", "model": "m",
                "schema": "jarvis.summarization", "schema_version": "1.0.0",
                "created_at": "2026-01-01T00:00:00Z",
            }
            (summaries_dir / f"{conv_id}_s{i:03d}.json").write_text(
                json.dumps(data), encoding="utf-8"
            )

        client = MagicMock()
        client.model = "test-model"
        client.generate.return_value = (_VALID_JSON_RESPONSE, False, "")
        client.parse_json_response.return_value = (json.loads(_VALID_JSON_RESPONSE), False, "")

        s = SegmentSummarizer(
            ollama_client=client,
            prompts_dir=prompts_dir,
            schema="jarvis.summarization",
            schema_version="1.0.0",
            context_window=3,
        )
        s.summarize_conversation_segments(
            segments_dir=segments_dir,
            conversation_id=conv_id,
            output_root=tmp_path / "OUTPUTS",
            from_segment=2,
        )

        prompt_used = client.generate.call_args_list[0][0][0]
        assert "Pre-existing summary 0." in prompt_used
        assert "Pre-existing summary 1." in prompt_used


# ---------------------------------------------------------------------------
# SummaryStore segment columns (schema v4)
# ---------------------------------------------------------------------------


class TestSummaryStoreSegmentColumns:
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

    def test_segment_columns_stored_and_retrieved(self, tmp_path):
        store = SummaryStore(str(tmp_path / "test.db"))
        output_data = self._minimal_output(
            source_kind="ai_chat_segment",
            segment_id="conv123_s005",
            segment_index=5,
            parent_conversation_id="conv123",
        )
        row_id = store.insert_summary(output_data)
        rows = store.get_by_ids([row_id])
        assert rows[0]["segment_id"] == "conv123_s005"
        assert rows[0]["segment_index"] == 5
        assert rows[0]["parent_conversation_id"] == "conv123"

    def test_get_segment_summaries_by_conversation(self, tmp_path):
        store = SummaryStore(str(tmp_path / "test.db"))
        for i in range(3):
            store.insert_summary(self._minimal_output(
                source_kind="ai_chat_segment",
                segment_id=f"conv_s{i:03d}",
                segment_index=i,
                parent_conversation_id="conv",
            ))
        store.insert_summary(self._minimal_output(source_kind="conversation"))

        results = store.get_segment_summaries_by_conversation("conv")
        assert len(results) == 3
        assert [r["segment_index"] for r in results] == [0, 1, 2]

    def test_conversation_summary_not_in_segment_query(self, tmp_path):
        store = SummaryStore(str(tmp_path / "test.db"))
        store.insert_summary(self._minimal_output(source_kind="conversation"))
        results = store.get_segment_summaries_by_conversation("conv")
        assert results == []

    def test_schema_version_is_five(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        SummaryStore(db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT value FROM _jarvis_meta WHERE key = 'schema_version'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert int(row[0]) == 5
