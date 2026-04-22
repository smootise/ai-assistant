"""Tests for Fragmenter — all Ollama calls mocked."""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from jarvis.fragmenter import Fragmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATEMENTS = [
    {"speaker": "user", "text": "We need an embedding model."},
    {"speaker": "assistant", "text": "qwen3-embedding is multilingual."},
    {"speaker": "user", "text": "What about latency?"},
    {"speaker": "assistant", "text": "It runs in ~50ms on CPU."},
]

_FRAGMENTS_RESPONSE = json.dumps({
    "fragments": [
        {
            "title": "Embedding model choice",
            "statements": _STATEMENTS[:2],
        },
        {
            "title": "Latency considerations",
            "statements": _STATEMENTS[2:],
        },
    ]
})


def _make_extract(seg_idx: int, conversation_id: str = "test-conv") -> Dict[str, Any]:
    return {
        "source_kind": "ai_chat_extract",
        "source_file": f"extract_{seg_idx:03d}.json",
        "segment_id": f"{conversation_id}_s{seg_idx:03d}",
        "segment_index": seg_idx,
        "parent_conversation_id": conversation_id,
        "statements": _STATEMENTS,
        "status": "ok",
        "model": "gemma4:31b",
        "latency_ms": 100,
        "created_at": "2026-04-21T00:00:00Z",
    }


def _make_fragmenter(prompts_dir: str) -> Fragmenter:
    client = MagicMock()
    client.model = "gemma4:31b"
    client.chat.return_value = (_FRAGMENTS_RESPONSE, False, "")
    client.parse_json_response.return_value = (
        json.loads(_FRAGMENTS_RESPONSE), False, ""
    )
    return Fragmenter(
        ollama_client=client,
        prompts_dir=prompts_dir,
        schema="jarvis.summarization",
        schema_version="1.0.0",
    )


@pytest.fixture()
def prompts_dir(tmp_path) -> str:
    (tmp_path / "fragment_extract.md").write_text(
        "Fragment these statements into coherent groups.\n---USER---\n{statements_text}", encoding="utf-8"
    )
    return str(tmp_path)


# ---------------------------------------------------------------------------
# fragment_extract
# ---------------------------------------------------------------------------


class TestFragmentExtract:
    def test_returns_correct_fragment_count(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        results = fragmenter.fragment_extract(
            extract_data=_make_extract(0),
            fragment_dir=tmp_path / "fragments",
        )
        assert len(results) == 2

    def test_output_fields(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        results = fragmenter.fragment_extract(
            extract_data=_make_extract(0),
            fragment_dir=tmp_path / "fragments",
        )
        _, frag0 = results[0]
        assert frag0["source_kind"] == "ai_chat_fragment"
        assert frag0["fragment_index"] == 0
        assert frag0["segment_index"] == 0
        assert frag0["segment_id"] == "test-conv_s000"
        assert frag0["parent_conversation_id"] == "test-conv"
        assert frag0["status"] == "ok"
        assert "text" in frag0
        assert "title" in frag0
        assert isinstance(frag0["statements"], list)

    def test_fragment_indices_sequential(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        results = fragmenter.fragment_extract(
            extract_data=_make_extract(0),
            fragment_dir=tmp_path / "fragments",
        )
        indices = [d["fragment_index"] for _, d in results]
        assert indices == list(range(len(results)))

    def test_files_written(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        fragment_dir = tmp_path / "fragments"
        fragmenter.fragment_extract(extract_data=_make_extract(2), fragment_dir=fragment_dir)
        assert (fragment_dir / "fragment_000.json").exists()
        assert (fragment_dir / "fragment_001.json").exists()
        assert (fragment_dir / "fragment_000.md").exists()

    def test_text_field_built_from_statements(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        results = fragmenter.fragment_extract(
            extract_data=_make_extract(0),
            fragment_dir=tmp_path / "fragments",
        )
        _, frag0 = results[0]
        assert "user:" in frag0["text"]
        assert "assistant:" in frag0["text"]

    def test_statements_in_prompt(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        fragmenter.fragment_extract(
            extract_data=_make_extract(0),
            fragment_dir=tmp_path / "fragments",
        )
        user_content = fragmenter._ollama.chat.call_args[0][1]
        assert "We need an embedding model." in user_content

    def test_empty_statements_returns_empty(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        extract = _make_extract(0)
        extract["statements"] = []
        results = fragmenter.fragment_extract(
            extract_data=extract,
            fragment_dir=tmp_path / "fragments",
        )
        assert results == []
        fragmenter._ollama.chat.assert_not_called()

    def test_degraded_sets_status(self, tmp_path):
        p = tmp_path / "prompts"
        p.mkdir()
        (p / "fragment_extract.md").write_text("Fragment statements.\n---USER---\n{statements_text}", encoding="utf-8")
        client = MagicMock()
        client.model = "gemma4:31b"
        client.chat.return_value = (_FRAGMENTS_RESPONSE, False, "")
        client.parse_json_response.return_value = (
            json.loads(_FRAGMENTS_RESPONSE), True, "Stripped code fences"
        )
        fragmenter = Fragmenter(
            ollama_client=client, prompts_dir=str(p),
            schema="jarvis.summarization", schema_version="1.0.0",
        )
        results = fragmenter.fragment_extract(
            extract_data=_make_extract(0),
            fragment_dir=tmp_path / "fragments",
        )
        assert all(d["status"] == "degraded" for _, d in results)


# ---------------------------------------------------------------------------
# fragment_conversation_extracts
# ---------------------------------------------------------------------------


class TestFragmentConversationExtracts:
    def _write_extracts(self, extract_dir: Path, n: int) -> None:
        extract_dir.mkdir(parents=True)
        for i in range(n):
            data = _make_extract(i)
            (extract_dir / f"extract_{i:03d}.json").write_text(
                json.dumps(data), encoding="utf-8"
            )

    def test_raises_if_extracts_missing(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        with pytest.raises(FileNotFoundError):
            fragmenter.fragment_conversation_extracts(
                conversation_id="conv",
                output_root=tmp_path / "OUTPUTS",
            )

    def test_fragments_all_extracts(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        output_root = tmp_path / "OUTPUTS"
        self._write_extracts(output_root / "conv" / "extracts", 3)
        results, skipped = fragmenter.fragment_conversation_extracts(
            conversation_id="conv",
            output_root=output_root,
        )
        # 3 segments × 2 fragments each
        assert len(results) == 6
        assert skipped == []
        assert fragmenter._ollama.chat.call_count == 3

    def test_skips_existing_without_force(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        output_root = tmp_path / "OUTPUTS"
        self._write_extracts(output_root / "conv" / "extracts", 2)

        seg0_dir = output_root / "conv" / "fragments" / "segment_000"
        seg0_dir.mkdir(parents=True)
        existing_frag = {
            "source_kind": "ai_chat_fragment", "fragment_index": 0, "segment_index": 0,
            "segment_id": "test-conv_s000", "parent_conversation_id": "test-conv",
            "text": "t", "title": "t", "statements": [], "status": "ok",
            "model": "m", "latency_ms": 0, "created_at": "2026-04-21T00:00:00Z",
        }
        (seg0_dir / "fragment_000.json").write_text(
            json.dumps(existing_frag), encoding="utf-8"
        )

        fragmenter.fragment_conversation_extracts(
            conversation_id="conv",
            output_root=output_root,
        )
        # Only segment 1 should be processed
        assert fragmenter._ollama.chat.call_count == 1

    def test_force_reprocesses_all(self, prompts_dir, tmp_path):
        fragmenter = _make_fragmenter(prompts_dir)
        output_root = tmp_path / "OUTPUTS"
        self._write_extracts(output_root / "conv" / "extracts", 2)

        seg0_dir = output_root / "conv" / "fragments" / "segment_000"
        seg0_dir.mkdir(parents=True)
        existing_frag = {
            "source_kind": "ai_chat_fragment", "fragment_index": 0, "segment_index": 0,
            "segment_id": "test-conv_s000", "parent_conversation_id": "test-conv",
            "text": "t", "title": "t", "statements": [], "status": "ok",
            "model": "m", "latency_ms": 0, "created_at": "2026-04-21T00:00:00Z",
        }
        (seg0_dir / "fragment_000.json").write_text(
            json.dumps(existing_frag), encoding="utf-8"
        )

        fragmenter.fragment_conversation_extracts(
            conversation_id="conv",
            output_root=output_root,
            force=True,
        )
        assert fragmenter._ollama.chat.call_count == 2
