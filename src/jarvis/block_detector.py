"""Deterministic preprocessing for segment text before LLM extraction.

Detects risky embedded blocks (code fences, large prompts, XML, JSON-schema,
long imperative sections) and builds a safe working view that replaces them
with labelled placeholders, preventing model drift during extraction.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


# Minimum size for a block to be archived (avoids noise on tiny snippets)
_MIN_CHARS = 400
_MIN_LINES = 6


@dataclass
class ArchivedBlock:
    block_id: str                                      # e.g. "ARCHIVED_BLOCK_1"
    block_kind: str                                    # see _BLOCK_KINDS
    speaker: Optional[str]                             # "user" | "assistant" | None
    line_count: int
    char_count: int
    raw_text: str                                      # kept for traceability + attempt-3 fallback
    archival_description: Optional[Dict[str, Any]] = field(default=None)


_BLOCK_KINDS = frozenset(
    ["fenced_code", "prompt_like", "xml_like", "json_schema_like", "imperative_block"]
)

# Tokens whose presence is evidence of a prompt-like block
_PROMPT_TOKENS = [
    "You are", "IMPORTANT:", "Return ONLY", "Do not", "Your job",
    "---USER---", "your task", "Your task",
]

_IMPERATIVE_STARTERS = re.compile(
    r"^(Do|Return|Use|Extract|Never|Always|Avoid|Include|Exclude|Output|Respond|Write|List)\b",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_blocks(segment_text: str) -> List[ArchivedBlock]:
    """Return all risky blocks detected in segment_text.

    Blocks are returned in the order they appear in the text.
    """
    # Track occupied char ranges to avoid double-counting overlapping detectors.
    # Each entry is (start, end) inclusive.
    occupied: List[Tuple[int, int]] = []
    candidates: List[Tuple[int, ArchivedBlock]] = []  # (start_pos, block)

    block_counter = [0]

    def _add(start: int, end: int, kind: str, raw: str) -> None:
        if _overlaps(start, end, occupied):
            return
        if len(raw) < _MIN_CHARS and raw.count("\n") + 1 < _MIN_LINES:
            return
        occupied.append((start, end))
        block_counter[0] += 1
        bid = f"ARCHIVED_BLOCK_{block_counter[0]}"
        speaker = _infer_speaker(segment_text, start)
        block = ArchivedBlock(
            block_id=bid,
            block_kind=kind,
            speaker=speaker,
            line_count=raw.count("\n") + 1,
            char_count=len(raw),
            raw_text=raw,
        )
        candidates.append((start, block))

    # --- Fenced code blocks (``` or ~~~) ---
    for m in re.finditer(r"(?m)^(`{3,}|~{3,})[^\n]*\n(.*?)^\1\s*$", segment_text, re.DOTALL):
        _add(m.start(), m.end(), "fenced_code", m.group(0))

    # --- JSON-schema-like: "type": "object" or "properties": { with ≥ 8 lines ---
    _json_schema_re = (
        r'(?s)(\{[^{}]{0,200}"type"\s*:\s*"object".*?\}'
        r'|\{[^{}]{0,200}"properties"\s*:\s*\{.*?\}\s*\})'
    )
    for m in re.finditer(_json_schema_re, segment_text):
        raw = m.group(0)
        if raw.count("\n") + 1 >= 8:
            _add(m.start(), m.end(), "json_schema_like", raw)

    # --- XML-like: contiguous lines where ≥ 60 % match XML tag pattern ---
    _detect_xml_blocks(segment_text, _add)

    # --- Prompt-like blocks ---
    _detect_prompt_blocks(segment_text, _add)

    # --- Long imperative blocks ---
    _detect_imperative_blocks(segment_text, _add)

    # Sort by position in the document
    candidates.sort(key=lambda t: t[0])
    # Re-assign sequential IDs in document order (counter may be out of order due to sort)
    blocks = []
    for idx, (_, blk) in enumerate(candidates, start=1):
        blk.block_id = f"ARCHIVED_BLOCK_{idx}"
        blocks.append(blk)

    return blocks


def build_working_view(
    segment_text: str,
    blocks: List[ArchivedBlock],
    mode: Literal["inventory", "archival"] = "inventory",
) -> str:
    """Build the extraction-safe working view of segment_text.

    Replaces each archived block with its placeholder and appends a
    deterministic inventory section.

    Args:
        segment_text: Original segment text.
        blocks: Detected archived blocks (from detect_blocks).
        mode: "inventory" — deterministic stats only.
              "archival" — stats + archival_description fields (if populated).

    Returns:
        Modified segment text with placeholders + inventory appended.
    """
    if not blocks:
        return segment_text

    working = segment_text
    # Replace in reverse order so positions stay valid
    for blk in reversed(blocks):
        working = working.replace(blk.raw_text, f"[{blk.block_id}]", 1)

    inventory_lines = ["", "---ARCHIVED BLOCKS---"]
    for blk in blocks:
        base = (
            f"[{blk.block_id}] kind={blk.block_kind} "
            f"speaker={blk.speaker or 'unknown'} "
            f"lines={blk.line_count} chars={blk.char_count}"
        )
        inventory_lines.append(base)
        if mode == "archival" and blk.archival_description:
            desc = blk.archival_description
            inventory_lines.append(
                f"  block_kind={desc.get('block_kind', '')} "
                f"is_instruction_like={desc.get('is_instruction_like', '')} "
                f"brief_description={desc.get('brief_description', '')}"
            )
            if desc.get("mentions"):
                inventory_lines.append(f"  mentions={', '.join(desc['mentions'])}")
            if desc.get("commands"):
                inventory_lines.append(f"  commands={', '.join(desc['commands'])}")
            if desc.get("paths"):
                inventory_lines.append(f"  paths={', '.join(desc['paths'])}")

    return working + "\n".join(inventory_lines)


def split_by_message(segment_text: str) -> List[Tuple[str, str]]:
    """Split segment_text into (speaker, message_text) pairs.

    Splits on lines that begin with "user:" or "assistant:" (case-insensitive).
    Each pair contains the speaker label and the message body (stripped).

    Returns:
        List of (speaker, message_text). Empty list if no speaker markers found.
    """
    pattern = re.compile(r"^(user|assistant)\s*:", re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(segment_text))
    if not matches:
        return []

    result: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        speaker = m.group(1).lower()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(segment_text)
        body = segment_text[body_start:body_end].strip()
        if body:
            result.append((speaker, body))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _overlaps(start: int, end: int, occupied: List[Tuple[int, int]]) -> bool:
    for s, e in occupied:
        if start < e and end > s:
            return True
    return False


def _infer_speaker(segment_text: str, block_start: int) -> Optional[str]:
    """Walk backwards from block_start to find the nearest speaker marker."""
    prefix = segment_text[:block_start]
    for line in reversed(prefix.splitlines()):
        stripped = line.strip().lower()
        if stripped.startswith("user:"):
            return "user"
        if stripped.startswith("assistant:"):
            return "assistant"
    return None


def _detect_xml_blocks(
    segment_text: str,
    add: Any,
) -> None:
    """Detect contiguous runs of lines that look like XML/HTML."""
    xml_tag = re.compile(r"</?[\w:-]+[\s>]")
    lines = segment_text.splitlines(keepends=True)
    run_start_char = 0
    run_lines: List[str] = []
    char_pos = 0

    def _flush(run_lines: List[str], run_start: int) -> None:
        raw = "".join(run_lines)
        xml_count = sum(1 for line in run_lines if xml_tag.search(line))
        if len(run_lines) >= 4 and xml_count / len(run_lines) >= 0.6:
            add(run_start, run_start + len(raw), "xml_like", raw)

    for line in lines:
        if xml_tag.search(line):
            if not run_lines:
                run_start_char = char_pos
            run_lines.append(line)
        else:
            if run_lines:
                _flush(run_lines, run_start_char)
                run_lines = []
        char_pos += len(line)

    if run_lines:
        _flush(run_lines, run_start_char)


def _detect_prompt_blocks(
    segment_text: str,
    add: Any,
) -> None:
    """Detect large blocks that look like embedded prompts."""
    lines = segment_text.splitlines(keepends=True)

    def _score(window: List[str]) -> int:
        text = "".join(window)
        token_hits = sum(1 for t in _PROMPT_TOKENS if t in text)
        heading_hits = sum(1 for line in window if re.match(r"^#{1,3}\s+\w", line))
        score = token_hits + (1 if heading_hits >= 3 else 0)
        return score

    char_pos = 0
    line_char_offsets = []
    for line in lines:
        line_char_offsets.append(char_pos)
        char_pos += len(line)

    n = len(lines)
    i = 0
    while i < n:
        # Try to extend a window starting at i
        j = i
        run_chars = 0
        while j < n:
            run_chars += len(lines[j])
            span_lines = lines[i: j + 1]
            if (j - i + 1 >= 20 or run_chars >= 1500) and _score(span_lines) >= 2:
                # Found a qualifying window; extend it as far as the score holds
                k = j + 1
                while k < n:
                    extended = lines[i: k + 1]
                    if _score(extended) >= 2:
                        k += 1
                    else:
                        break
                raw = "".join(lines[i:k])
                add(line_char_offsets[i], line_char_offsets[i] + len(raw), "prompt_like", raw)
                i = k  # skip past this window
                break
            j += 1
        else:
            i += 1


def _detect_imperative_blocks(
    segment_text: str,
    add: Any,
) -> None:
    """Detect contiguous regions of ≥ 15 lines with ≥ 5 imperative-starting lines."""
    lines = segment_text.splitlines(keepends=True)
    n = len(lines)
    line_char_offsets = []
    char_pos = 0
    for line in lines:
        line_char_offsets.append(char_pos)
        char_pos += len(line)

    i = 0
    while i < n:
        # Collect a run of lines where many start with imperative verbs
        j = i
        imp_count = 0
        while j < n:
            if _IMPERATIVE_STARTERS.match(lines[j]):
                imp_count += 1
            run_len = j - i + 1
            if run_len >= 15 and imp_count >= 5:
                # Extend while still meeting threshold
                k = j + 1
                while k < n:
                    if _IMPERATIVE_STARTERS.match(lines[k]):
                        imp_count += 1
                    k += 1
                    run_len2 = k - i
                    if imp_count / run_len2 < 5 / 15:
                        break
                raw = "".join(lines[i:k])
                add(line_char_offsets[i], line_char_offsets[i] + len(raw), "imperative_block", raw)
                i = k
                break
            j += 1
        else:
            i += 1
