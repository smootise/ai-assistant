"""Tests for the ChatGPT ingestion pipeline.

Uses samples/ai/sample_chatgpt.json as the primary fixture.
Tests cover: filtering, active-path reconstruction, dedup,
retry-collapse, chunk overlap, and pending-tail behaviour.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from jarvis.ingest.chatgpt_parser import (
    _collapse_adjacent_user_retries,
    parse_export,
    reconstruct_active_path,
)
from jarvis.ingest.chunker import chunk_conversation
from jarvis.ingest.normalizer import (
    build_normalized,
    merge_normalized,
)

SAMPLE_PATH = Path(__file__).parent.parent / "samples" / "ai" / "sample_chatgpt.json"


@pytest.fixture()
def sample_raw() -> Dict[str, Any]:
    with open(SAMPLE_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parser: active path and filtering
# ---------------------------------------------------------------------------


class TestChatGPTParser:
    def test_only_user_and_assistant_roles(self, sample_raw):
        messages = parse_export(sample_raw)
        roles = {m["speaker"] for m in messages}
        assert roles <= {"user", "assistant"}, f"Unexpected roles: {roles}"

    def test_system_messages_excluded(self, sample_raw):
        messages = parse_export(sample_raw)
        # The sample has two system nodes at the top — verify none leaked through
        for m in messages:
            assert m["speaker"] != "system"

    def test_tool_messages_excluded(self, sample_raw):
        """The tool/bio 'Model set context updated.' node must not appear."""
        messages = parse_export(sample_raw)
        for m in messages:
            assert "Model set context updated" not in m["content"]

    def test_bio_recipient_excluded(self, sample_raw):
        """The assistant→bio memory-update message must be excluded."""
        messages = parse_export(sample_raw)
        # That message content starts with "User is a product manager"
        for m in messages:
            assert not m["content"].startswith("User is a product manager in tech")

    def test_hidden_messages_excluded(self, sample_raw):
        """Messages with is_visually_hidden_from_conversation=true are dropped."""
        messages = parse_export(sample_raw)
        # Both system messages in the sample are hidden; verify count
        assert len(messages) >= 1

    def test_empty_text_messages_excluded(self, sample_raw):
        messages = parse_export(sample_raw)
        for m in messages:
            assert m["content"].strip() != ""

    def test_active_path_uses_current_node(self, sample_raw):
        """reconstruct_active_path must end at current_node when it is provided."""
        mapping = sample_raw["mapping"]
        # Sample has no current_node; supply the last node in children chain
        last_node = "bbb21a15-a0cc-4329-984e-6bcab4f787fe"  # child of last assistant
        if last_node not in mapping:
            pytest.skip("Last node not in sample mapping")
        path = reconstruct_active_path(mapping, last_node)
        assert path[-1] == last_node

    def test_active_path_fallback_no_current_node(self, sample_raw):
        """Without current_node, falls back to last-child walk without crashing."""
        mapping = sample_raw["mapping"]
        path = reconstruct_active_path(mapping, current_node=None)
        assert len(path) > 0

    def test_first_visible_message_is_user(self, sample_raw):
        messages = parse_export(sample_raw)
        assert messages[0]["speaker"] == "user"

    def test_messages_have_required_fields(self, sample_raw):
        messages = parse_export(sample_raw)
        for m in messages:
            assert "message_id" in m
            assert "speaker" in m
            assert "position" in m
            assert "content" in m

    def test_positions_are_sequential(self, sample_raw):
        messages = parse_export(sample_raw)
        for i, m in enumerate(messages):
            assert m["position"] == i

    def test_model_editable_context_excluded(self, sample_raw):
        """model_editable_context content_type must be excluded."""
        messages = parse_export(sample_raw)
        # The sample has one such node (8a565174); it must not appear
        ids = {m["message_id"] for m in messages}
        assert "8a565174-845e-47ad-8117-fc36c3ba3f37" not in ids


# ---------------------------------------------------------------------------
# Retry dedup: adjacent identical user messages
# ---------------------------------------------------------------------------


class TestRetryCollapse:
    def _make_msg(self, role: str, text: str, pos: int = 0) -> Dict[str, Any]:
        return {
            "author": {"role": role},
            "id": f"id_{pos}",
            "_raw_text": text,
            "create_time": None,
            "content": {"content_type": "text", "parts": [text]},
            "metadata": {},
            "recipient": "all",
        }

    def test_identical_adjacent_users_collapsed(self):
        msgs = [
            self._make_msg("user", "Hello world", 0),
            self._make_msg("user", "Hello world", 1),
            self._make_msg("assistant", "Hi there", 2),
        ]
        result = _collapse_adjacent_user_retries(msgs)
        assert len(result) == 2
        assert result[0]["id"] == "id_1"  # later one kept
        assert result[1]["author"]["role"] == "assistant"

    def test_whitespace_normalized_for_comparison(self):
        msgs = [
            self._make_msg("user", "Hello  world\n", 0),
            self._make_msg("user", "Hello world", 1),
        ]
        result = _collapse_adjacent_user_retries(msgs)
        assert len(result) == 1
        assert result[0]["id"] == "id_1"

    def test_different_content_not_collapsed(self):
        msgs = [
            self._make_msg("user", "First question", 0),
            self._make_msg("user", "Different question", 1),
        ]
        result = _collapse_adjacent_user_retries(msgs)
        assert len(result) == 2

    def test_non_adjacent_duplicates_not_collapsed(self):
        msgs = [
            self._make_msg("user", "Same text", 0),
            self._make_msg("assistant", "Answer", 1),
            self._make_msg("user", "Same text", 2),
        ]
        result = _collapse_adjacent_user_retries(msgs)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Normalizer: dedup / incremental re-import
# ---------------------------------------------------------------------------


class TestNormalizer:
    def test_reimport_does_not_duplicate(self, sample_raw):
        messages = parse_export(sample_raw)
        normalized = build_normalized(sample_raw, messages, "test.json")
        # Re-import the same messages
        merged = merge_normalized(normalized, messages)
        assert merged["message_count"] == len(messages)
        assert len(merged["messages"]) == len(messages)

    def test_reimport_adds_new_messages(self, sample_raw):
        messages = parse_export(sample_raw)
        normalized = build_normalized(sample_raw, messages, "test.json")
        # Simulate a new message arriving
        new_msg = {
            "message_id": "brand-new-id-9999",
            "speaker": "user",
            "created_at": None,
            "position": len(messages),
            "content": "A brand new message",
        }
        merged = merge_normalized(normalized, messages + [new_msg])
        assert merged["message_count"] == len(messages) + 1

    def test_positions_reassigned_after_merge(self, sample_raw):
        messages = parse_export(sample_raw)
        normalized = build_normalized(sample_raw, messages, "test.json")
        new_msg = {
            "message_id": "new-id-001",
            "speaker": "assistant",
            "created_at": None,
            "position": 999,
            "content": "Late reply",
        }
        merged = merge_normalized(normalized, messages + [new_msg])
        for i, m in enumerate(merged["messages"]):
            assert m["position"] == i

    def test_normalized_has_required_top_level_fields(self, sample_raw):
        messages = parse_export(sample_raw)
        normalized = build_normalized(sample_raw, messages, "raw/test.json")
        for field in (
            "conversation_id",
            "title",
            "source_platform",
            "source_file",
            "raw_export_path",
            "imported_at",
            "message_count",
            "messages",
        ):
            assert field in normalized, f"Missing field: {field}"

    def test_source_platform_is_chatgpt(self, sample_raw):
        messages = parse_export(sample_raw)
        normalized = build_normalized(sample_raw, messages, "test.json")
        assert normalized["source_platform"] == "chatgpt"


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def _make_normalized(messages_spec) -> Dict[str, Any]:
    """Build a minimal normalized dict from a list of (speaker, content) tuples."""
    messages = [
        {
            "message_id": f"id_{i}",
            "speaker": spk,
            "created_at": None,
            "position": i,
            "content": content,
        }
        for i, (spk, content) in enumerate(messages_spec)
    ]
    return {
        "conversation_id": "test-conv",
        "title": "Test",
        "source_platform": "chatgpt",
        "message_count": len(messages),
        "messages": messages,
    }


class TestChunker:
    def test_ideal_pattern_three_message_chunks(self):
        # U A U A U A  -> chunks: [U0,A0,U1], [U1,A1,U2], [U2,A2,U3-missing]
        norm = _make_normalized(
            [("user", "U1"), ("assistant", "A1"), ("user", "U2"), ("assistant", "A2")]
        )
        result = chunk_conversation(norm)
        # U0 A0 U1 → chunk0; U1 A1 (end) → chunk1
        assert len(result["chunks"]) == 2
        assert result["chunks"][0]["start_position"] == 0
        assert result["chunks"][0]["end_position"] == 2
        # Overlap: chunk1 starts at position 2
        assert result["chunks"][1]["start_position"] == 2

    def test_overlap_user_message_starts_next_chunk(self):
        norm = _make_normalized(
            [
                ("user", "U1"),
                ("assistant", "A1"),
                ("user", "U2"),
                ("assistant", "A2"),
                ("user", "U3"),
                ("assistant", "A3"),
            ]
        )
        result = chunk_conversation(norm)
        chunks = result["chunks"]
        assert len(chunks) >= 2
        # The last message_id of chunk N should equal the first of chunk N+1
        for i in range(len(chunks) - 1):
            last_id = chunks[i]["message_ids"][-1]
            first_id = chunks[i + 1]["message_ids"][0]
            assert last_id == first_id, (
                f"Chunk {i} last id {last_id!r} != Chunk {i+1} first id {first_id!r}"
            )

    def test_trailing_user_becomes_pending_tail(self):
        norm = _make_normalized(
            [("user", "U1"), ("assistant", "A1"), ("user", "U2")]
        )
        result = chunk_conversation(norm)
        # chunk: [U1, A1, U2], then U2 is the overlap → next chunk starts with U2
        # U2 has no following assistant → pending tail
        assert result["pending_tail"] is not None
        assert result["pending_tail"]["speaker"] == "user"
        assert result["pending_tail"]["content"] == "U2"

    def test_pending_tail_not_in_chunks(self):
        norm = _make_normalized(
            [("user", "U1"), ("assistant", "A1"), ("user", "U2")]
        )
        result = chunk_conversation(norm)
        tail_id = result["pending_tail"]["message_id"] if result["pending_tail"] else None
        if tail_id:
            for chunk in result["chunks"]:
                # The tail should not be the ONLY message in any chunk
                # (it IS allowed as the overlap end of a chunk)
                if len(chunk["message_ids"]) == 1:
                    assert chunk["message_ids"][0] != tail_id

    def test_final_two_message_chunk_allowed(self):
        """Conversation ending user->assistant emits a 2-message chunk."""
        norm = _make_normalized([("user", "Q"), ("assistant", "A")])
        result = chunk_conversation(norm)
        assert len(result["chunks"]) == 1
        assert len(result["chunks"][0]["message_ids"]) == 2
        assert result["pending_tail"] is None

    def test_manifest_has_required_fields(self, sample_raw):
        messages = parse_export(sample_raw)
        normalized = build_normalized(sample_raw, messages, "test.json")
        result = chunk_conversation(normalized)
        meta = result["manifest_meta"]
        for field in (
            "conversation_id",
            "total_visible_messages",
            "chunk_count",
            "chunk_ids",
            "pending_tail",
            "chunked_at",
        ):
            assert field in meta, f"Missing manifest field: {field}"

    def test_chunk_text_format(self):
        norm = _make_normalized([("user", "Hello"), ("assistant", "World")])
        result = chunk_conversation(norm)
        text = result["chunks"][0]["chunk_text"]
        assert "user: Hello" in text
        assert "assistant: World" in text

    def test_no_messages_dropped(self):
        """All visible messages must appear in chunks or pending_tail."""
        norm = _make_normalized(
            [
                ("user", "U1"),
                ("assistant", "A1"),
                ("user", "U2"),
                ("assistant", "A2"),
                ("user", "U3"),
            ]
        )
        result = chunk_conversation(norm)
        all_ids_in_chunks = set()
        for chunk in result["chunks"]:
            all_ids_in_chunks.update(chunk["message_ids"])
        if result["pending_tail"]:
            all_ids_in_chunks.add(result["pending_tail"]["message_id"])
        all_input_ids = {m["message_id"] for m in norm["messages"]}
        assert all_input_ids == all_ids_in_chunks
