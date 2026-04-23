"""Tests for block_detector — all pure/deterministic, no I/O, no LLM."""

from jarvis.block_detector import (
    ArchivedBlock,
    build_working_view,
    detect_blocks,
    split_by_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fenced_block(n_lines: int = 30, lang: str = "python") -> str:
    code = "\n".join(f"    x_{i} = {i}" for i in range(n_lines))
    return f"```{lang}\n{code}\n```"


# ---------------------------------------------------------------------------
# detect_blocks
# ---------------------------------------------------------------------------

class TestDetectBlocks:
    def test_no_blocks_on_clean_segment(self):
        text = "user: What is the capital of France?\n\nassistant: Paris."
        assert detect_blocks(text) == []

    def test_detects_fenced_code_block(self):
        code = _fenced_block(30)
        text = f"user: Here is my code:\n\n{code}\n\nassistant: I see."
        blocks = detect_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].block_kind == "fenced_code"
        assert blocks[0].line_count >= 30
        assert blocks[0].char_count > 0
        assert blocks[0].char_count <= len(code) + 2  # regex match may include trailing newline

    def test_fenced_block_id_is_archived_block_1(self):
        text = f"user: check this\n\n{_fenced_block(30)}\n\nassistant: ok"
        blocks = detect_blocks(text)
        assert blocks[0].block_id == "ARCHIVED_BLOCK_1"

    def test_fenced_block_speaker_inferred(self):
        text = f"user: see below\n\n{_fenced_block(30)}"
        blocks = detect_blocks(text)
        assert blocks[0].speaker == "user"

    def test_long_markdown_answer_not_archived(self):
        # A long structured assistant answer should NOT be archived — it's legitimate content
        answer = "\n".join(
            ["## Section one", "Some explanation here.", ""] * 10
            + ["Use this approach.", "Do not skip steps.", "Return the result."] * 5
        )
        text = f"user: how do I do this?\n\nassistant: {answer}\n\nuser: thanks"
        blocks = detect_blocks(text)
        assert blocks == []

    def test_sub_threshold_block_ignored(self):
        # Fenced block that is below 400 chars AND below 6 lines
        tiny = "```python\nx = 1\n```"
        text = f"user: small snippet\n\n{tiny}\n\nassistant: yes"
        blocks = detect_blocks(text)
        assert blocks == []

    def test_multiple_blocks_sequential_ids(self):
        code1 = _fenced_block(30)
        code2 = _fenced_block(35, lang="bash")
        text = f"user: first\n\n{code1}\n\nassistant: ok\n\nuser: second\n\n{code2}"
        blocks = detect_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].block_id == "ARCHIVED_BLOCK_1"
        assert blocks[1].block_id == "ARCHIVED_BLOCK_2"

    def test_raw_text_preserved(self):
        code = _fenced_block(30)
        text = f"user: see\n\n{code}"
        blocks = detect_blocks(text)
        assert blocks[0].raw_text == code


# ---------------------------------------------------------------------------
# build_working_view
# ---------------------------------------------------------------------------

class TestBuildWorkingView:
    def test_no_blocks_returns_original(self):
        text = "user: hello\n\nassistant: hi"
        assert build_working_view(text, []) == text

    def test_placeholder_replaces_raw_block(self):
        code = _fenced_block(30)
        text = f"user: code below\n\n{code}\n\nassistant: ok"
        blocks = detect_blocks(text)
        view = build_working_view(text, blocks)
        assert "[ARCHIVED_BLOCK_1]" in view
        # Raw code should not appear in the view
        assert "x_0 = 0" not in view

    def test_inventory_section_appended(self):
        code = _fenced_block(30)
        text = f"user: code below\n\n{code}\n\nassistant: ok"
        blocks = detect_blocks(text)
        view = build_working_view(text, blocks, mode="inventory")
        assert "---ARCHIVED BLOCKS---" in view
        assert "kind=fenced_code" in view
        assert "lines=" in view
        assert "chars=" in view

    def test_archival_mode_includes_description(self):
        code = _fenced_block(30)
        text = f"user: code below\n\n{code}"
        blocks = detect_blocks(text)
        # Manually inject an archival description
        blocks[0].archival_description = {
            "block_kind": "code",
            "is_instruction_like": False,
            "brief_description": "Python variable assignments",
            "mentions": ["Python"],
            "commands": [],
            "paths": [],
        }
        view = build_working_view(text, blocks, mode="archival")
        assert "Python variable assignments" in view

    def test_inventory_mode_excludes_archival_description(self):
        code = _fenced_block(30)
        text = f"user: code below\n\n{code}"
        blocks = detect_blocks(text)
        blocks[0].archival_description = {
            "block_kind": "code",
            "is_instruction_like": False,
            "brief_description": "Should not appear in inventory mode",
        }
        view = build_working_view(text, blocks, mode="inventory")
        assert "Should not appear in inventory mode" not in view


# ---------------------------------------------------------------------------
# split_by_message
# ---------------------------------------------------------------------------

class TestSplitByMessage:
    def test_simple_alternation(self):
        text = "user: Hello\n\nassistant: Hi there"
        pairs = split_by_message(text)
        assert len(pairs) == 2
        assert pairs[0] == ("user", "Hello")
        assert pairs[1] == ("assistant", "Hi there")

    def test_multi_turn(self):
        text = (
            "user: Question one\n\n"
            "assistant: Answer one\n\n"
            "user: Question two\n\n"
            "assistant: Answer two"
        )
        pairs = split_by_message(text)
        assert len(pairs) == 4
        assert pairs[0][0] == "user"
        assert pairs[1][0] == "assistant"

    def test_no_markers_returns_empty(self):
        text = "Some text without speaker markers."
        assert split_by_message(text) == []

    def test_case_insensitive(self):
        text = "User: Hello\n\nAssistant: World"
        pairs = split_by_message(text)
        assert pairs[0][0] == "user"
        assert pairs[1][0] == "assistant"

    def test_message_body_stripped(self):
        text = "user:   hello   \n\nassistant: world"
        pairs = split_by_message(text)
        assert pairs[0][1] == "hello"
