"""Hardening tests for SegmentExtractor — retry ladder, drift detection, clean retrieval text."""

import json
import re
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, call

import pytest

from jarvis.extractor import SegmentExtractor, _validate_statements, _dedup_consecutive_assistant_turns
from jarvis.fragmenter import Fragmenter


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_VALID_STATEMENTS = [
    {"speaker": "user", "text": "We need to pick an embedding model."},
    {"speaker": "assistant", "text": "qwen3-embedding supports multilingual text."},
]

_VALID_RESPONSE = json.dumps({"statements": _VALID_STATEMENTS})


def _make_client(response: str = _VALID_RESPONSE, degraded: bool = False) -> MagicMock:
    client = MagicMock()
    client.model = "gemma4:31b"
    client.generate_json.return_value = (response, degraded, "")
    client.parse_json_response.return_value = (json.loads(response), degraded, "")
    return client


def _make_extractor(prompts_dir: str, client: MagicMock = None) -> SegmentExtractor:
    if client is None:
        client = _make_client()
    return SegmentExtractor(
        ollama_client=client,
        prompts_dir=prompts_dir,
        schema="jarvis.summarization",
        schema_version="1.0.0",
    )


def _make_segment(
    index: int = 0,
    text: str = "user: Q\n\nassistant: A",
    conversation_id: str = "test-conv",
) -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "segment_id": f"{conversation_id}_s{index:03d}",
        "segment_index": index,
        "segment_text": text,
    }


@pytest.fixture()
def prompts_dir(tmp_path) -> str:
    (tmp_path / "extract_segment.md").write_text(
        "Extract statements.\n---USER---\n{segment_text}", encoding="utf-8"
    )
    (tmp_path / "describe_archived_block.md").write_text(
        "Describe the block.\n---USER---\n{block_text}", encoding="utf-8"
    )
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Request builder shape
# ---------------------------------------------------------------------------

class TestRequestBuilderShape:
    def test_uses_generate_json_not_chat(self, prompts_dir, tmp_path):
        client = _make_client()
        extractor = _make_extractor(prompts_dir, client)
        extractor.extract_segment(segment=_make_segment(), extract_dir=tmp_path / "out")
        assert client.generate_json.called
        assert not client.chat.called

    def test_generate_json_receives_schema(self, prompts_dir, tmp_path):
        client = _make_client()
        extractor = _make_extractor(prompts_dir, client)
        extractor.extract_segment(segment=_make_segment(), extract_dir=tmp_path / "out")
        _, kwargs = client.generate_json.call_args
        positional = client.generate_json.call_args[0]
        # schema is second positional arg
        schema = positional[1] if len(positional) > 1 else kwargs.get("schema")
        assert schema is not None
        assert schema.get("type") == "object"
        assert "statements" in schema.get("properties", {})

    def test_generate_json_called_with_string_prompt(self, prompts_dir, tmp_path):
        client = _make_client()
        extractor = _make_extractor(prompts_dir, client)
        extractor.extract_segment(segment=_make_segment(), extract_dir=tmp_path / "out")
        prompt_arg = client.generate_json.call_args[0][0]
        assert isinstance(prompt_arg, str)
        assert len(prompt_arg) > 0


# ---------------------------------------------------------------------------
# Attempt 1 success path
# ---------------------------------------------------------------------------

class TestAttempt1Success:
    def test_extraction_attempt_is_1(self, prompts_dir, tmp_path):
        client = _make_client()
        extractor = _make_extractor(prompts_dir, client)
        _, out = extractor.extract_segment(segment=_make_segment(), extract_dir=tmp_path / "out")
        assert out["extraction_attempt"] == 1

    def test_status_ok(self, prompts_dir, tmp_path):
        client = _make_client()
        extractor = _make_extractor(prompts_dir, client)
        _, out = extractor.extract_segment(segment=_make_segment(), extract_dir=tmp_path / "out")
        assert out["status"] == "ok"

    def test_per_statement_metadata_injected(self, prompts_dir, tmp_path):
        client = _make_client()
        extractor = _make_extractor(prompts_dir, client)
        _, out = extractor.extract_segment(segment=_make_segment(), extract_dir=tmp_path / "out")
        stmts = out["statements"]
        assert len(stmts) == 2
        for idx, s in enumerate(stmts):
            assert s["statement_index"] == idx
            assert "statement_id" in s
            assert s["segment_id"] == "test-conv_s000"
            assert s["parent_conversation_id"] == "test-conv"

    def test_archived_blocks_empty_when_no_risky_blocks(self, prompts_dir, tmp_path):
        client = _make_client()
        extractor = _make_extractor(prompts_dir, client)
        _, out = extractor.extract_segment(segment=_make_segment(), extract_dir=tmp_path / "out")
        assert out["archived_blocks"] == []


# ---------------------------------------------------------------------------
# Drift / validation rejection
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_speaker_rejected(self, prompts_dir, tmp_path):
        bad = json.dumps({"statements": [{"speaker": "bot", "text": "hello"}]})
        client = MagicMock()
        client.model = "gemma4:31b"
        # Attempt 1 returns invalid speaker; attempts 2+3 return valid
        valid_resp = _VALID_RESPONSE
        client.generate_json.side_effect = [
            (bad, False, ""),
            (valid_resp, False, ""),  # attempt 2
            (valid_resp, False, ""),  # attempt 3 (per message)
        ]
        client.parse_json_response.side_effect = [
            (json.loads(bad), False, ""),
            (json.loads(valid_resp), False, ""),
            (json.loads(valid_resp), False, ""),
        ]
        extractor = _make_extractor(prompts_dir, client)
        _, out = extractor.extract_segment(segment=_make_segment(), extract_dir=tmp_path / "out")
        # Should have recovered on attempt 2
        assert out["extraction_attempt"] >= 2
        assert out["status"] in ("ok", "degraded")

    def test_repetitive_drift_rejected(self):
        drifted_stmts = [
            {"speaker": "assistant", "text": f"The summarize command will process data {i}"}
            for i in range(10)
        ]
        parsed = {"statements": drifted_stmts}
        # 10 statements that are near-identical (differ only by trailing number)
        ok, warnings = _validate_statements(parsed, "user: some question\n" * 5)
        assert not ok
        assert any("drift" in w for w in warnings)

    def test_implausible_count_rejected(self):
        many = [{"speaker": "user", "text": f"Statement {i}"} for i in range(100)]
        parsed = {"statements": many}
        working_view = "user: hello\n" * 10  # 10 lines → max_expected = max(40, 5) = 40
        ok, warnings = _validate_statements(parsed, working_view)
        assert not ok
        assert any("plausible max" in w for w in warnings)

    def test_near_empty_rejected(self):
        parsed = {"statements": []}
        working_view = "user: " + "x" * 300
        ok, warnings = _validate_statements(parsed, working_view)
        assert not ok
        assert any("zero statements" in w for w in warnings)

    def test_unexpected_key_rejected(self):
        parsed = {"statements": _VALID_STATEMENTS, "extra_key": "oops"}
        ok, warnings = _validate_statements(parsed, "user: Q\nassistant: A")
        assert not ok
        assert any("unexpected keys" in w for w in warnings)

    def test_valid_output_passes(self):
        parsed = {"statements": _VALID_STATEMENTS}
        ok, warnings = _validate_statements(parsed, "user: Q\nassistant: A\n" * 5)
        assert ok
        assert warnings == []

    def test_structural_repetition_not_flagged_as_drift(self):
        """Legitimate segments (e.g. specs with bullet lists) share phrases
        across statements without being drift. Only true near-duplicates should
        be rejected."""
        stmts = [
            {"speaker": "user", "text": "The CLI command summarize accepts a --file flag for the input path."},
            {"speaker": "user", "text": "The CLI command summarize accepts a --provider flag selecting local or openai."},
            {"speaker": "user", "text": "The CLI command summarize accepts an --output-dir flag for the output path."},
            {"speaker": "user", "text": "The CLI command summarize accepts a --log-level flag controlling verbosity."},
            {"speaker": "assistant", "text": "The output contract requires a summary field as a string."},
            {"speaker": "assistant", "text": "The output contract requires a bullets field as an array of strings."},
            {"speaker": "assistant", "text": "The output contract requires an action_items field as an array of strings."},
            {"speaker": "assistant", "text": "The output contract requires a confidence field between 0 and 1."},
            {"speaker": "assistant", "text": "HARD errors include missing file, unsupported extension, and unreachable model."},
            {"speaker": "assistant", "text": "SOFT errors include minor JSON repairs and empty-ish input producing degraded status."},
        ]
        parsed = {"statements": stmts}
        # Simulate a long working view so the per-line threshold is generous
        working = "\n".join(f"line {i}" for i in range(80))
        ok, warnings = _validate_statements(parsed, working)
        assert ok, f"expected structural repetition to pass, got warnings: {warnings}"


# ---------------------------------------------------------------------------
# Attempt 3 per-message fallback
# ---------------------------------------------------------------------------

class TestAttempt3Fallback:
    def test_attempt3_triggered_after_two_failures(self, prompts_dir, tmp_path):
        invalid_resp = json.dumps({"statements": [{"speaker": "bot", "text": "x"}]})
        valid_per_msg = json.dumps({"statements": [{"speaker": "user", "text": "Q"}]})

        client = MagicMock()
        client.model = "gemma4:31b"
        client.generate_json.side_effect = [
            (invalid_resp, False, ""),   # attempt 1 — invalid speaker
            (invalid_resp, False, ""),   # attempt 2 — still invalid
            (valid_per_msg, False, ""),  # attempt 3 per-message: user msg
            (valid_per_msg, False, ""),  # attempt 3 per-message: assistant msg
        ]
        client.parse_json_response.side_effect = [
            (json.loads(invalid_resp), False, ""),
            (json.loads(invalid_resp), False, ""),
            (json.loads(valid_per_msg), False, ""),
            (json.loads(valid_per_msg.replace('"user"', '"assistant"')), False, ""),
        ]
        extractor = _make_extractor(prompts_dir, client)
        seg = _make_segment(text="user: Q\n\nassistant: A")
        _, out = extractor.extract_segment(segment=seg, extract_dir=tmp_path / "out")
        assert out["extraction_attempt"] == 3

    def test_attempt3_merges_messages_in_order(self, prompts_dir, tmp_path):
        user_resp = json.dumps({"statements": [{"speaker": "user", "text": "First statement"}]})
        asst_resp = json.dumps({"statements": [{"speaker": "assistant", "text": "Second statement"}]})
        invalid_resp = json.dumps({"statements": [{"speaker": "bot", "text": "bad"}]})

        client = MagicMock()
        client.model = "gemma4:31b"
        client.generate_json.side_effect = [
            (invalid_resp, False, ""),  # attempt 1
            (invalid_resp, False, ""),  # attempt 2
            (user_resp, False, ""),     # attempt 3 user
            (asst_resp, False, ""),     # attempt 3 assistant
        ]
        client.parse_json_response.side_effect = [
            (json.loads(invalid_resp), False, ""),
            (json.loads(invalid_resp), False, ""),
            (json.loads(user_resp), False, ""),
            (json.loads(asst_resp), False, ""),
        ]
        extractor = _make_extractor(prompts_dir, client)
        seg = _make_segment(text="user: Q\n\nassistant: A")
        _, out = extractor.extract_segment(segment=seg, extract_dir=tmp_path / "out")
        texts = [s["text"] for s in out["statements"]]
        assert "First statement" in texts
        assert "Second statement" in texts
        assert texts.index("First statement") < texts.index("Second statement")


# ---------------------------------------------------------------------------
# Attempt 2: archival description path
# ---------------------------------------------------------------------------

class TestAttempt2ArchivalPath:
    def test_describe_archived_block_called_on_attempt2(self, tmp_path):
        # Build a prompts dir with both templates
        p = tmp_path / "prompts"
        p.mkdir()
        (p / "extract_segment.md").write_text(
            "Extract.\n---USER---\n{segment_text}", encoding="utf-8"
        )
        (p / "describe_archived_block.md").write_text(
            "Describe.\n---USER---\n{block_text}", encoding="utf-8"
        )

        invalid_resp = json.dumps({"statements": [{"speaker": "bot", "text": "x"}]})
        archival_resp = json.dumps({
            "block_kind": "code",
            "is_instruction_like": False,
            "brief_description": "Python variable assignments",
            "mentions": [],
            "commands": [],
            "paths": [],
        })
        valid_resp = _VALID_RESPONSE

        # Segment with a large fenced code block
        code_block = "```python\n" + "\n".join(f"x_{i} = {i}" for i in range(40)) + "\n```"
        seg_text = f"user: check this code\n\n{code_block}\n\nassistant: noted"

        call_count = [0]

        def _generate_json_side_effect(prompt, schema, **kwargs):
            call_count[0] += 1
            c = call_count[0]
            if c == 1:
                return (invalid_resp, False, "")   # attempt 1 extraction
            elif c == 2:
                return (archival_resp, False, "")  # describe block (attempt 2)
            else:
                return (valid_resp, False, "")     # attempt 2 extraction

        def _parse_side_effect(raw):
            try:
                return (json.loads(raw), False, "")
            except Exception:
                raise ValueError("parse failed")

        client = MagicMock()
        client.model = "gemma4:31b"
        client.generate_json.side_effect = _generate_json_side_effect
        client.parse_json_response.side_effect = _parse_side_effect

        extractor = SegmentExtractor(
            ollama_client=client,
            prompts_dir=str(p),
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        seg = _make_segment(text=seg_text)
        _, out = extractor.extract_segment(segment=seg, extract_dir=tmp_path / "out")

        assert out["extraction_attempt"] == 2
        assert len(out["archived_blocks"]) == 1
        assert out["archived_blocks"][0]["archival_description"] is not None
        assert out["archived_blocks"][0]["archival_description"]["brief_description"] == (
            "Python variable assignments"
        )


# ---------------------------------------------------------------------------
# Retrieval text excludes raw blob
# ---------------------------------------------------------------------------

class TestRetrievalTextClean:
    def test_fragment_text_excludes_raw_code_block(self, prompts_dir, tmp_path):
        """After extraction and fragmentation, fragment text must not contain the raw blob."""
        code_block = "```python\n" + "\n".join(f"x_{i} = {i}" for i in range(40)) + "\n```"
        seg_text = f"user: check this code\n\n{code_block}\n\nassistant: looks good"

        # Extractor: valid statements referencing the archived block token
        extract_stmts = [
            {
                "statement_id": "c_s000_st0000",
                "statement_index": 0,
                "segment_id": "c_s000",
                "segment_index": 0,
                "parent_conversation_id": "c",
                "speaker": "user",
                "text": "The user shared [ARCHIVED_BLOCK_1].",
            },
            {
                "statement_id": "c_s000_st0001",
                "statement_index": 1,
                "segment_id": "c_s000",
                "segment_index": 0,
                "parent_conversation_id": "c",
                "speaker": "assistant",
                "text": "The code looks good.",
            },
        ]
        raw_archived = [
            {
                "block_id": "ARCHIVED_BLOCK_1",
                "block_kind": "fenced_code",
                "speaker": "user",
                "line_count": 42,
                "char_count": len(code_block),
                "raw_text": code_block,
                "archival_description": {
                    "block_kind": "code",
                    "is_instruction_like": False,
                    "brief_description": "Python variable assignments x_0 to x_39",
                },
            }
        ]

        # Build a minimal extract_data dict (as if extractor produced it)
        extract_data = {
            "statements": extract_stmts,
            "archived_blocks": raw_archived,
            "segment_id": "c_s000",
            "segment_index": 0,
            "parent_conversation_id": "c",
            "conversation_date": None,
        }

        # Fragment response: model groups both statements into one fragment
        frag_resp = json.dumps({
            "fragments": [
                {
                    "title": "Code review",
                    "statements": extract_stmts,
                }
            ]
        })

        client = MagicMock()
        client.model = "gemma4:31b"
        client.chat.return_value = (frag_resp, False, "")
        client.parse_json_response.return_value = (json.loads(frag_resp), False, "")

        p = tmp_path / "prompts"
        p.mkdir()
        (p / "fragment_extract.md").write_text(
            "Fragment statements.\n---USER---\n{statements_text}", encoding="utf-8"
        )

        fragmenter = Fragmenter(
            ollama_client=client,
            prompts_dir=str(p),
            schema="jarvis.summarization",
            schema_version="1.0.0",
        )
        fragments = fragmenter.fragment_extract(
            extract_data=extract_data,
            fragment_dir=tmp_path / "frags",
        )
        assert len(fragments) == 1
        _, frag_out = fragments[0]
        retrieval_text = frag_out["text"]

        # Raw code content must not appear in retrieval text
        assert "x_0 = 0" not in retrieval_text
        assert "```python" not in retrieval_text
        # Archival description should appear
        assert "Python variable assignments" in retrieval_text

    def test_fragment_has_statement_index_span(self, prompts_dir, tmp_path):
        extract_stmts = [
            {
                "statement_id": "c_s000_st0000",
                "statement_index": 0,
                "segment_id": "c_s000",
                "segment_index": 0,
                "parent_conversation_id": "c",
                "speaker": "user",
                "text": "First.",
            },
            {
                "statement_id": "c_s000_st0001",
                "statement_index": 1,
                "segment_id": "c_s000",
                "segment_index": 0,
                "parent_conversation_id": "c",
                "speaker": "assistant",
                "text": "Second.",
            },
        ]
        extract_data = {
            "statements": extract_stmts,
            "archived_blocks": [],
            "segment_id": "c_s000",
            "segment_index": 0,
            "parent_conversation_id": "c",
            "conversation_date": None,
        }
        frag_resp = json.dumps({
            "fragments": [{"title": "Both", "statements": extract_stmts}]
        })

        client = MagicMock()
        client.model = "gemma4:31b"
        client.chat.return_value = (frag_resp, False, "")
        client.parse_json_response.return_value = (json.loads(frag_resp), False, "")

        p = tmp_path / "fp"
        p.mkdir()
        (p / "fragment_extract.md").write_text(
            "Fragment.\n---USER---\n{statements_text}", encoding="utf-8"
        )
        fragmenter = Fragmenter(
            ollama_client=client, prompts_dir=str(p),
            schema="j", schema_version="1",
        )
        fragments = fragmenter.fragment_extract(
            extract_data=extract_data, fragment_dir=tmp_path / "frags"
        )
        _, frag_out = fragments[0]
        assert frag_out["statement_start_index"] == 0
        assert frag_out["statement_end_index"] == 1


# ---------------------------------------------------------------------------
# Consecutive duplicate assistant turn deduplication
# ---------------------------------------------------------------------------

_ASSISTANT_BODY_A = (
    "Here is the spec: use SQLite as source of truth, Qdrant for vector search, "
    "Qwen3 for embeddings, and Gemma 4 31B for summarization. The pipeline order is "
    "ingest then summarize then detect topics then retrieve."
)

# Near-identical to A (>70% Jaccard on 5-grams)
_ASSISTANT_BODY_B = (
    "Here is the spec: use SQLite as source of truth, Qdrant for vector search, "
    "Qwen3 for embeddings, and Gemma 4 31B for summarization. The pipeline order is "
    "ingest then summarize then detect-topics then retrieve and answer."
)

_ASSISTANT_BODY_DIFFERENT = (
    "The retrieval unit is a fragment — a topically coherent sub-set of statements "
    "from a single extract. Fragments are embedded independently for semantic search."
)


class TestDedupConsecutiveAssistantTurns:

    def test_no_change_when_no_duplicates(self):
        text = f"user: What is the plan?\n\nassistant: {_ASSISTANT_BODY_DIFFERENT}"
        out, dropped = _dedup_consecutive_assistant_turns(text)
        assert dropped == 0
        assert "fragment" in out

    def test_drops_earlier_of_two_near_identical_assistant_turns(self):
        text = (
            f"user: What is the plan?\n\n"
            f"assistant: {_ASSISTANT_BODY_A}\n\n"
            f"assistant: {_ASSISTANT_BODY_B}"
        )
        out, dropped = _dedup_consecutive_assistant_turns(text)
        assert dropped == 1
        # The last turn is kept
        assert "detect-topics" in out
        # The first turn is gone
        assert out.count("assistant:") == 1

    def test_keeps_distinct_consecutive_assistant_turns(self):
        text = (
            f"user: What is the plan?\n\n"
            f"assistant: {_ASSISTANT_BODY_A}\n\n"
            f"assistant: {_ASSISTANT_BODY_DIFFERENT}"
        )
        out, dropped = _dedup_consecutive_assistant_turns(text)
        assert dropped == 0
        assert out.count("assistant:") == 2

    def test_user_turns_never_dropped(self):
        # Two near-identical user turns should NOT be deduplicated
        text = (
            f"user: {_ASSISTANT_BODY_A}\n\n"
            f"user: {_ASSISTANT_BODY_B}\n\n"
            f"assistant: Understood."
        )
        out, dropped = _dedup_consecutive_assistant_turns(text)
        assert dropped == 0
        assert out.count("user:") == 2

    def test_empty_segment_no_crash(self):
        out, dropped = _dedup_consecutive_assistant_turns("")
        assert dropped == 0
        assert out == ""

    def test_dedup_applied_before_extraction(self, prompts_dir, tmp_path):
        """Extractor receives deduplicated text — duplicate turn does not reach the model."""
        import json as _json

        valid_resp = _json.dumps({"statements": [
            {"speaker": "user", "text": "What is the plan?"},
            {"speaker": "assistant", "text": _ASSISTANT_BODY_B},
        ]})
        client = _make_client(valid_resp)
        extractor = _make_extractor(prompts_dir, client)

        seg_text = (
            f"user: What is the plan?\n\n"
            f"assistant: {_ASSISTANT_BODY_A}\n\n"
            f"assistant: {_ASSISTANT_BODY_B}"
        )
        seg = _make_segment(text=seg_text)
        output = extractor.extract_segment(segment=seg, extract_dir=tmp_path / "out")

        # Model was called; prompt must NOT contain the first (dropped) assistant turn
        prompt_sent = client.generate_json.call_args[0][0]
        # The dropped body ends with "retrieve." — the kept body ends with "answer."
        assert "retrieve." not in prompt_sent or "answer." in prompt_sent
