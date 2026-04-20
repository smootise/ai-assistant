"""Tests for SegmentExtractor — all Ollama calls mocked."""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from jarvis.extractor import SegmentExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATEMENTS = [
    {"speaker": "user", "text": "We need to pick an embedding model."},
    {"speaker": "assistant", "text": "qwen3-embedding supports multilingual text."},
]

_VALID_RESPONSE = json.dumps({"statements": _STATEMENTS})


def _make_segment(index: int, conversation_id: str = "test-conv") -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "segment_id": f"{conversation_id}_s{index:03d}",
        "segment_index": index,
        "segment_text": f"user: Q{index}\n\nassistant: A{index}",
    }


def _make_extractor(prompts_dir: str) -> SegmentExtractor:
    client = MagicMock()
    client.model = "gemma4:31b"
    client.generate.return_value = (_VALID_RESPONSE, False, "")
    client.parse_json_response.return_value = (json.loads(_VALID_RESPONSE), False, "")
    return SegmentExtractor(
        ollama_client=client,
        prompts_dir=prompts_dir,
        schema="jarvis.summarization",
        schema_version="1.0.0",
    )


@pytest.fixture()
def prompts_dir(tmp_path) -> str:
    (tmp_path / "extract_segment.md").write_text(
        "Extract statements from:\n\n{segment_text}", encoding="utf-8"
    )
    return str(tmp_path)


# ---------------------------------------------------------------------------
# extract_segment
# ---------------------------------------------------------------------------


class TestExtractSegment:
    def test_output_fields(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        segment = _make_segment(0)
        _, output_data = extractor.extract_segment(
            segment=segment,
            extract_dir=tmp_path / "extracts",
        )
        assert output_data["source_kind"] == "ai_chat_extract"
        assert output_data["segment_id"] == "test-conv_s000"
        assert output_data["segment_index"] == 0
        assert output_data["parent_conversation_id"] == "test-conv"
        assert output_data["status"] == "ok"
        assert isinstance(output_data["statements"], list)

    def test_statements_stored(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        _, output_data = extractor.extract_segment(
            segment=_make_segment(0),
            extract_dir=tmp_path / "extracts",
        )
        assert len(output_data["statements"]) == 2
        assert output_data["statements"][0]["speaker"] == "user"

    def test_files_written(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        extract_dir = tmp_path / "extracts"
        extractor.extract_segment(segment=_make_segment(3), extract_dir=extract_dir)
        assert (extract_dir / "extract_003.json").exists()
        assert (extract_dir / "extract_003.md").exists()

    def test_segment_text_in_prompt(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        segment = _make_segment(0)
        extractor.extract_segment(segment=segment, extract_dir=tmp_path / "extracts")
        prompt_used = extractor._ollama.generate.call_args[0][0]
        assert segment["segment_text"] in prompt_used

    def test_degraded_response_sets_status(self, tmp_path):
        p = tmp_path / "prompts"
        p.mkdir()
        (p / "extract_segment.md").write_text("{segment_text}", encoding="utf-8")
        client = MagicMock()
        client.model = "gemma4:31b"
        client.generate.return_value = (_VALID_RESPONSE, False, "")
        client.parse_json_response.return_value = (
            json.loads(_VALID_RESPONSE), True, "Stripped code fences"
        )
        extractor = SegmentExtractor(
            ollama_client=client,
            prompts_dir=str(p),
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        _, output_data = extractor.extract_segment(
            segment=_make_segment(0),
            extract_dir=tmp_path / "extracts",
        )
        assert output_data["status"] == "degraded"
        assert len(output_data.get("warnings", [])) > 0


# ---------------------------------------------------------------------------
# extract_conversation_segments
# ---------------------------------------------------------------------------


class TestExtractConversationSegments:
    def _write_segments(self, segments_dir: Path, n: int) -> None:
        segments_dir.mkdir(parents=True)
        for i in range(n):
            seg = _make_segment(i)
            (segments_dir / f"segment_{i:03d}.json").write_text(
                json.dumps(seg), encoding="utf-8"
            )

    def test_raises_if_segments_dir_missing(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        with pytest.raises(FileNotFoundError):
            extractor.extract_conversation_segments(
                segments_dir=tmp_path / "nonexistent",
                conversation_id="x",
                output_root=tmp_path,
            )

    def test_processes_all_segments(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        segments_dir = tmp_path / "segments"
        self._write_segments(segments_dir, 3)
        results = extractor.extract_conversation_segments(
            segments_dir=segments_dir,
            conversation_id="conv",
            output_root=tmp_path / "OUTPUTS",
        )
        assert len(results) == 3
        assert extractor._ollama.generate.call_count == 3

    def test_skips_existing_without_force(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        segments_dir = tmp_path / "segments"
        self._write_segments(segments_dir, 2)

        output_root = tmp_path / "OUTPUTS"
        extract_dir = output_root / "conv" / "extracts"
        extract_dir.mkdir(parents=True)
        existing = {"source_kind": "ai_chat_extract", "segment_index": 0,
                    "segment_id": "conv_s000", "parent_conversation_id": "conv",
                    "statements": [], "status": "ok", "latency_ms": 0}
        (extract_dir / "extract_000.json").write_text(
            json.dumps(existing), encoding="utf-8"
        )

        results = extractor.extract_conversation_segments(
            segments_dir=segments_dir,
            conversation_id="conv",
            output_root=output_root,
        )
        assert len(results) == 2
        assert extractor._ollama.generate.call_count == 1

    def test_force_overwrites(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        segments_dir = tmp_path / "segments"
        self._write_segments(segments_dir, 2)

        output_root = tmp_path / "OUTPUTS"
        extract_dir = output_root / "conv" / "extracts"
        extract_dir.mkdir(parents=True)
        existing = {"source_kind": "ai_chat_extract", "segment_index": 0,
                    "segment_id": "conv_s000", "parent_conversation_id": "conv",
                    "statements": [], "status": "ok", "latency_ms": 0}
        (extract_dir / "extract_000.json").write_text(
            json.dumps(existing), encoding="utf-8"
        )

        extractor.extract_conversation_segments(
            segments_dir=segments_dir,
            conversation_id="conv",
            output_root=output_root,
            force=True,
        )
        assert extractor._ollama.generate.call_count == 2

    def test_skips_pending_tail(self, prompts_dir, tmp_path):
        extractor = _make_extractor(prompts_dir)
        segments_dir = tmp_path / "segments"
        self._write_segments(segments_dir, 1)
        (segments_dir / "pending_tail.json").write_text("{}", encoding="utf-8")

        results = extractor.extract_conversation_segments(
            segments_dir=segments_dir,
            conversation_id="conv",
            output_root=tmp_path / "OUTPUTS",
        )
        assert len(results) == 1
        assert extractor._ollama.generate.call_count == 1
