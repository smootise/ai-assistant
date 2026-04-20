"""Tests for TopicDetector — all Ollama and embedding calls mocked."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from jarvis.topic_detector import TopicDetector, _cosine_similarity
from jarvis.store import SummaryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment_summary(segment_index: int, summary: str = "summary text") -> Dict[str, Any]:
    return {
        "source_kind": "ai_chat_segment",
        "segment_id": f"686c5e2b_s{segment_index:03d}",
        "segment_index": segment_index,
        "parent_conversation_id": "test-conv-id",
        "summary": summary,
        "bullets": ["bullet"],
        "action_items": [],
        "confidence": 0.8,
        "provider": "local",
        "model": "gemma4:31b",
        "schema": "jarvis.summarization",
        "schema_version": "1.0.0",
        "status": "ok",
        "created_at": "2026-04-17T00:00:00Z",
        "source_file": f"686c5e2b_s{segment_index:03d}.json",
        "latency_ms": 1000,
    }


def _make_detector(
    threshold: float = 0.65,
    vector_store=None,
    prompts_dir: str = "prompts",
) -> TopicDetector:
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    ollama = MagicMock()
    ollama.model = "gemma4:31b"
    ollama.generate.return_value = (
        '{"summary": "topic summary", "bullets": [], "action_items": [], "confidence": 0.8}',
        False,
        "",
    )
    ollama.parse_json_response.return_value = (
        {
            "summary": "topic summary",
            "bullets": [],
            "action_items": [],
            "confidence": 0.8,
            "status": "ok",
            "created_at": "2026-04-17T00:00:00Z",
            "latency_ms": 500,
        },
        False,
        "",
    )
    return TopicDetector(
        embedder=embedder,
        ollama_client=ollama,
        prompts_dir=prompts_dir,
        schema="jarvis.summarization",
        schema_version="1.0.0",
        threshold=threshold,
        vector_store=vector_store,
    )


# ---------------------------------------------------------------------------
# TestCosineSimilarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        zero = [0.0, 0.0, 0.0]
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(zero, v) == 0.0
        assert _cosine_similarity(v, zero) == 0.0


# ---------------------------------------------------------------------------
# TestDetectTopics
# ---------------------------------------------------------------------------

class TestDetectTopics:
    def _high_sim_vectors(self, n: int) -> List[List[float]]:
        return [[1.0, 0.0] for _ in range(n)]

    def test_no_boundary_single_topic(self):
        detector = _make_detector(threshold=0.65)
        segments = [_make_segment_summary(i) for i in range(4)]
        vectors = self._high_sim_vectors(4)
        topics, sims = detector.detect_topics(segments, vectors)
        assert len(topics) == 1
        assert len(topics[0]) == 4

    def test_one_boundary_two_topics(self):
        detector = _make_detector(threshold=0.65)
        segments = [_make_segment_summary(i) for i in range(4)]
        vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        topics, sims = detector.detect_topics(segments, vectors)
        assert len(topics) == 2
        assert len(topics[0]) == 2
        assert len(topics[1]) == 2

    def test_boundary_count_matches_expected(self):
        detector = _make_detector(threshold=0.65)
        segments = [_make_segment_summary(i) for i in range(5)]
        vectors = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        topics, _ = detector.detect_topics(segments, vectors)
        assert len(topics) == 3

    def test_single_segment_returns_one_topic(self):
        detector = _make_detector()
        segments = [_make_segment_summary(0)]
        topics, sims = detector.detect_topics(segments, [[1.0, 0.0]])
        assert len(topics) == 1
        assert sims == []

    def test_similarities_list_length(self):
        detector = _make_detector()
        n = 5
        segments = [_make_segment_summary(i) for i in range(n)]
        vectors = self._high_sim_vectors(n)
        _, sims = detector.detect_topics(segments, vectors)
        assert len(sims) == n - 1


# ---------------------------------------------------------------------------
# TestLoadEmbeddings
# ---------------------------------------------------------------------------

class TestLoadEmbeddings:
    def test_always_embeds_on_the_fly(self):
        detector = _make_detector(vector_store=None)
        segments = [_make_segment_summary(i) for i in range(3)]
        detector._load_embeddings("conv-id", segments)
        assert detector._embedder.embed.call_count == 3

    def test_embeds_summary_text_only(self):
        detector = _make_detector(vector_store=None)
        segment = _make_segment_summary(0, summary="my summary text")
        detector._load_embeddings("conv-id", [segment])
        call_arg = detector._embedder.embed.call_args[0][0]
        assert call_arg == "my summary text"

    def test_qdrant_not_queried_even_when_available(self):
        vector_store = MagicMock()
        detector = _make_detector(vector_store=vector_store)
        segments = [_make_segment_summary(0)]
        detector._load_embeddings("conv-id", segments)
        vector_store.get_by_conversation.assert_not_called()
        detector._embedder.embed.assert_called_once()


# ---------------------------------------------------------------------------
# TestSummarizeTopic
# ---------------------------------------------------------------------------

class TestSummarizeTopic:
    def test_output_fields(self, tmp_path):
        detector = _make_detector(prompts_dir=str(tmp_path))
        (tmp_path / "summarize_topic.md").write_text(
            "You are JARVIS.", encoding="utf-8"
        )
        topic_dir = tmp_path / "topics"
        segments = [_make_segment_summary(0), _make_segment_summary(1)]
        _, output_data = detector.summarize_topic(
            topic_segments=segments,
            topic_index=0,
            conversation_id="test-conv",
            topic_summaries_dir=topic_dir,
        )
        assert output_data["source_kind"] == "ai_chat_topic"
        assert output_data["topic_index"] == 0
        assert output_data["parent_conversation_id"] == "test-conv"
        assert "topic_segment_range" in output_data

    def test_prompt_contains_all_segment_summaries(self, tmp_path):
        detector = _make_detector(prompts_dir=str(tmp_path))
        (tmp_path / "summarize_topic.md").write_text("TEMPLATE", encoding="utf-8")
        topic_dir = tmp_path / "topics"
        segments = [
            _make_segment_summary(0, summary="alpha summary"),
            _make_segment_summary(1, summary="beta summary"),
        ]
        detector.summarize_topic(
            topic_segments=segments,
            topic_index=0,
            conversation_id="conv",
            topic_summaries_dir=topic_dir,
        )
        call_args = detector._ollama.generate.call_args[0][0]
        assert "alpha summary" in call_args
        assert "beta summary" in call_args

    def test_degraded_json_handled(self, tmp_path):
        detector = _make_detector(prompts_dir=str(tmp_path))
        (tmp_path / "summarize_topic.md").write_text("TEMPLATE", encoding="utf-8")
        detector._ollama.parse_json_response.return_value = (
            {
                "summary": "fallback",
                "bullets": [],
                "action_items": [],
                "confidence": 0.3,
                "status": "degraded",
                "created_at": "2026-04-17T00:00:00Z",
                "latency_ms": 100,
            },
            True,
            "Model output required cleanup",
        )
        topic_dir = tmp_path / "topics"
        _, output_data = detector.summarize_topic(
            topic_segments=[_make_segment_summary(0)],
            topic_index=0,
            conversation_id="conv",
            topic_summaries_dir=topic_dir,
        )
        assert output_data["status"] == "degraded"


# ---------------------------------------------------------------------------
# TestDetectAndSummarize
# ---------------------------------------------------------------------------

class TestDetectAndSummarize:
    def _write_segment_summaries(self, output_root: Path, conv_id: str, n: int) -> Path:
        d = output_root / conv_id / "segment_summaries"
        d.mkdir(parents=True)
        for i in range(n):
            seg = _make_segment_summary(i)
            (d / f"{seg['segment_id']}.json").write_text(
                json.dumps(seg), encoding="utf-8"
            )
        return d

    def test_raises_if_segment_summaries_missing(self, tmp_path):
        detector = _make_detector()
        with pytest.raises(FileNotFoundError):
            detector.detect_and_summarize("missing-conv", tmp_path)

    def test_dry_run_skips_llm(self, tmp_path):
        detector = _make_detector(prompts_dir=str(tmp_path))
        (tmp_path / "summarize_topic.md").write_text("TEMPLATE", encoding="utf-8")
        self._write_segment_summaries(tmp_path, "conv-id", 3)
        results = detector.detect_and_summarize(
            "conv-id", tmp_path, dry_run=True
        )
        assert results == []
        detector._ollama.generate.assert_not_called()

    def test_end_to_end_two_topics(self, tmp_path):
        detector = _make_detector(prompts_dir=str(tmp_path), threshold=0.65)
        (tmp_path / "summarize_topic.md").write_text("TEMPLATE", encoding="utf-8")
        self._write_segment_summaries(tmp_path, "conv-id", 4)

        high = [1.0, 0.0]
        low = [0.0, 1.0]
        detector._embedder.embed.side_effect = [high, high, low, low]

        results = detector.detect_and_summarize("conv-id", tmp_path)
        assert len(results) == 2
        topic_dir = tmp_path / "conv-id" / "topic_summaries"
        assert (topic_dir / "topic_000.json").exists()
        assert (topic_dir / "topic_001.json").exists()

    def test_stale_warning_logged(self, tmp_path, caplog):
        detector = _make_detector(prompts_dir=str(tmp_path))
        (tmp_path / "summarize_topic.md").write_text("TEMPLATE", encoding="utf-8")
        self._write_segment_summaries(tmp_path, "conv-id", 2)
        topic_dir = tmp_path / "conv-id" / "topic_summaries"
        topic_dir.mkdir(parents=True)
        (topic_dir / "topic_000.json").write_text("{}", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            detector.detect_and_summarize("conv-id", tmp_path)

        assert any("stale" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# TestSummaryStoreTopicColumns
# ---------------------------------------------------------------------------

class TestSummaryStoreTopicColumns:
    def _make_topic_record(self, topic_index: int = 0) -> Dict[str, Any]:
        return {
            "source_file": f"topic_{topic_index:03d}.json",
            "source_kind": "ai_chat_topic",
            "provider": "local",
            "model": "gemma4:31b",
            "schema": "jarvis.summarization",
            "schema_version": "1.0.0",
            "status": "ok",
            "confidence": 0.85,
            "summary": "Topic summary here",
            "bullets": ["decision A"],
            "action_items": [],
            "created_at": "2026-04-17T00:00:00Z",
            "topic_index": topic_index,
            "topic_segment_range": "s000-s004",
            "parent_conversation_id": "test-conv",
        }

    def test_topic_columns_stored_and_retrieved(self, tmp_path):
        store = SummaryStore(db_path=str(tmp_path / "test.db"))
        record = self._make_topic_record(0)
        row_id = store.insert_summary(record)
        rows = store.get_topic_summaries_by_conversation("test-conv")
        assert len(rows) == 1
        assert rows[0]["topic_index"] == 0
        assert rows[0]["topic_segment_range"] == "s000-s004"
        assert rows[0]["id"] == row_id

    def test_get_topic_summaries_by_conversation(self, tmp_path):
        store = SummaryStore(db_path=str(tmp_path / "test.db"))
        for i in range(3):
            store.insert_summary(self._make_topic_record(i))
        segment_record = {
            "source_file": "segment.json",
            "source_kind": "ai_chat_segment",
            "provider": "local",
            "model": "gemma4:31b",
            "schema": "jarvis.summarization",
            "schema_version": "1.0.0",
            "status": "ok",
            "confidence": 0.8,
            "summary": "segment summary",
            "bullets": [],
            "action_items": [],
            "created_at": "2026-04-17T00:00:00Z",
            "parent_conversation_id": "test-conv",
            "segment_id": "s000",
            "segment_index": 0,
        }
        store.insert_summary(segment_record)
        rows = store.get_topic_summaries_by_conversation("test-conv")
        assert len(rows) == 3
        assert [r["topic_index"] for r in rows] == [0, 1, 2]
