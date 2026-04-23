"""Segment extractor for JARVIS.

Extracts all informational content from a conversation segment as a clean
list of attributed statements (speaker + text). This is the first step in
the extract→fragment retrieval pipeline.

Hardening: segments with embedded code/prompt blocks are preprocessed
deterministically. The LLM sees placeholders + an inventory; raw blobs are
never passed to the model. A 3-step retry ladder uses materially different
requests at each attempt.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jarvis.block_detector import ArchivedBlock, build_working_view, detect_blocks, split_by_message
from jarvis.ollama import OllamaClient


logger = logging.getLogger(__name__)

# JSON schema enforced via Ollama's structured output
_STATEMENTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["user", "assistant"]},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
        }
    },
    "required": ["statements"],
}

# Schema for the archival block description LLM call
_ARCHIVAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "block_kind": {"type": "string"},
        "is_instruction_like": {"type": "boolean"},
        "brief_description": {"type": "string"},
        "mentions": {"type": "array", "items": {"type": "string"}},
        "commands": {"type": "array", "items": {"type": "string"}},
        "paths": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["block_kind", "is_instruction_like", "brief_description"],
}


class SegmentExtractor:
    """Extracts attributed statements from conversation segments."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        prompts_dir: str,
        schema: str,
        schema_version: str,
    ):
        self._ollama = ollama_client
        self._prompts_dir = Path(prompts_dir)
        self._schema = schema
        self._schema_version = schema_version

    def extract_conversation_segments(
        self,
        segments_dir: Path,
        conversation_id: str,
        output_root: Path,
        from_segment: int = 0,
        to_segment: Optional[int] = None,
        force: bool = False,
        retries: int = 1,
    ) -> List[Tuple[Path, Dict[str, Any]]]:
        """Extract all segments for a conversation.

        Resume-safe: skips segments whose extract file already exists unless
        force=True.

        Args:
            segments_dir: Directory containing segment_NNN.json files.
            conversation_id: Parent conversation ID.
            output_root: Root output directory (e.g. OUTPUTS/).
            from_segment: First segment index to process (inclusive).
            to_segment: Last segment index to process (inclusive). None = last.
            force: If True, overwrite existing extract files.
            retries: Unused legacy param (ladder has fixed 3 steps); kept for
                     CLI compat.

        Returns:
            List of (extract_dir, output_data) tuples.

        Raises:
            FileNotFoundError: If segments_dir does not exist.
        """
        if not segments_dir.exists():
            raise FileNotFoundError(f"Segments directory not found: {segments_dir}")

        extract_dir = output_root / conversation_id / "extracts"

        segment_files = sorted(
            f for f in segments_dir.glob("segment_*.json")
            if f.name != "pending_tail.json"
        )

        segments = []
        for path in segment_files:
            with open(path, encoding="utf-8") as f:
                segments.append(json.load(f))

        segments.sort(key=lambda s: s["segment_index"])

        effective_to = to_segment if to_segment is not None else segments[-1]["segment_index"]
        selected = [s for s in segments if from_segment <= s["segment_index"] <= effective_to]

        if not selected:
            logger.warning(
                f"No segments in range [{from_segment}, {effective_to}] — nothing to extract."
            )
            return []

        results: List[Tuple[Path, Dict[str, Any]]] = []
        for segment in selected:
            seg_idx = segment["segment_index"]
            existing_path = extract_dir / f"extract_{seg_idx:03d}.json"

            if existing_path.exists() and not force:
                with open(existing_path, encoding="utf-8") as f:
                    output_data = json.load(f)
                if output_data.get("status") in ("skipped", "partial"):
                    logger.info(
                        f"Re-processing segment {seg_idx} ({segment['segment_id']}) "
                        f"— previous extract has status '{output_data['status']}'"
                    )
                else:
                    logger.info(
                        f"Skipping segment {seg_idx} ({segment['segment_id']}) "
                        f"— extract already exists"
                    )
                    results.append((extract_dir, output_data))
                    continue

            logger.info(
                f"Extracting segment {seg_idx} / {segments[-1]['segment_index']} "
                f"({segment['segment_id']})"
            )
            _, output_data = self.extract_segment(
                segment=segment,
                extract_dir=extract_dir,
            )
            results.append((extract_dir, output_data))

        logger.info(
            f"Extracted {len(results)} segments for conversation {conversation_id}"
        )
        return results

    def extract_segment(
        self,
        segment: Dict[str, Any],
        extract_dir: Path,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Extract statements from a single segment using the 3-step retry ladder.

        Attempt 1: generate_json with placeholders + deterministic inventory.
        Attempt 2: generate_json with placeholders + archival block descriptions
                   (triggers a describe_archived_block LLM call per risky block).
        Attempt 3: per-message extraction merged in order.

        Args:
            segment: Segment dict (from segment_NNN.json).
            extract_dir: Directory to write extract artifacts.

        Returns:
            Tuple of (extract_dir, output_data).
        """
        start_time = time.time()
        segment_text = segment["segment_text"]
        segment_id = segment["segment_id"]

        # Detect risky blocks once; used across all attempts
        blocks = detect_blocks(segment_text)
        if blocks:
            logger.info(
                f"Segment {segment_id}: {len(blocks)} archived block(s) detected "
                f"({', '.join(b.block_kind for b in blocks)})"
            )

        statements: List[Dict[str, Any]] = []
        archived_blocks: List[Dict[str, Any]] = []
        is_degraded = False
        is_partial = False
        warnings: List[str] = []
        extraction_attempt = 0
        skip_reason: Optional[str] = None

        # --- Attempt 1: inventory mode ---
        extraction_attempt = 1
        working_view = build_working_view(segment_text, blocks, mode="inventory")
        prompt = self._build_prompt(working_view)
        try:
            raw, gen_degraded, gen_warning = self._ollama.generate_json(prompt, _STATEMENTS_SCHEMA)
        except RuntimeError as e:
            skip_reason = f"Attempt 1 timeout: {e}"
            logger.error(f"Segment {segment_id} timed out on attempt 1 — skipping")
            raw = ""

        if not skip_reason:
            try:
                parsed, parse_degraded, parse_warning = self._ollama.parse_json_response(raw)
                is_degraded = gen_degraded or parse_degraded
                if parse_warning:
                    warnings.append(parse_warning)
                valid, val_warnings = _validate_statements(parsed, working_view)
                if valid:
                    statements = parsed.get("statements", [])
                else:
                    warnings.extend(val_warnings)
                    logger.warning(
                        f"Segment {segment_id} attempt 1 validation failed: "
                        + "; ".join(val_warnings)
                    )
                    # Fall through to attempt 2
            except ValueError as e:
                warnings.append(f"Attempt 1 parse failed: {e}")
                logger.warning(f"Segment {segment_id} attempt 1 parse failed — trying attempt 2")

        # --- Attempt 2: archival descriptions ---
        if not skip_reason and not statements:
            extraction_attempt = 2
            if blocks:
                blocks = self._describe_blocks(blocks, segment_id)
            working_view2 = build_working_view(segment_text, blocks, mode="archival")
            prompt2 = self._build_prompt(working_view2)
            try:
                raw2, gen_degraded2, gen_warning2 = self._ollama.generate_json(
                    prompt2, _STATEMENTS_SCHEMA
                )
            except RuntimeError as e:
                skip_reason = f"Attempt 2 timeout: {e}"
                logger.error(f"Segment {segment_id} timed out on attempt 2 — skipping")
                raw2 = ""

            if not skip_reason:
                try:
                    parsed2, parse_degraded2, parse_warning2 = self._ollama.parse_json_response(
                        raw2
                    )
                    is_degraded = is_degraded or gen_degraded2 or parse_degraded2
                    if parse_warning2:
                        warnings.append(parse_warning2)
                    valid2, val_warnings2 = _validate_statements(parsed2, working_view2)
                    if valid2:
                        statements = parsed2.get("statements", [])
                    else:
                        warnings.extend(val_warnings2)
                        logger.warning(
                            f"Segment {segment_id} attempt 2 validation failed: "
                            + "; ".join(val_warnings2)
                        )
                except ValueError as e:
                    warnings.append(f"Attempt 2 parse failed: {e}")
                    logger.warning(
                        f"Segment {segment_id} attempt 2 parse failed — trying attempt 3"
                    )

        # --- Attempt 3: per-message fallback ---
        if not skip_reason and not statements:
            extraction_attempt = 3
            messages = split_by_message(segment_text)
            if not messages:
                # No speaker markers — treat entire text as one message with unknown speaker
                messages = [("user", segment_text)]

            merged: List[Dict[str, Any]] = []
            attempt3_failed = False
            for speaker, msg_text in messages:
                msg_blocks = detect_blocks(msg_text)
                msg_view = build_working_view(msg_text, msg_blocks, mode="inventory")
                msg_prompt = self._build_prompt(msg_view)
                try:
                    raw3, _, _ = self._ollama.generate_json(msg_prompt, _STATEMENTS_SCHEMA)
                except RuntimeError as e:
                    warnings.append(f"Attempt 3 timeout for {speaker} message: {e}")
                    attempt3_failed = True
                    break
                try:
                    parsed3, _, _ = self._ollama.parse_json_response(raw3)
                    msg_stmts = parsed3.get("statements", []) if isinstance(parsed3, dict) else []
                    # Normalise speaker to the message's known speaker when unambiguous
                    for s in msg_stmts:
                        if s.get("speaker") not in ("user", "assistant"):
                            s["speaker"] = speaker
                    merged.extend(msg_stmts)
                except ValueError as e:
                    warnings.append(f"Attempt 3 parse failed for {speaker} message: {e}")
                    attempt3_failed = True
                    break

            if not attempt3_failed and merged:
                valid3, val_warnings3 = _validate_statements(
                    {"statements": merged}, segment_text
                )
                if valid3:
                    statements = merged
                else:
                    warnings.extend(val_warnings3)
                    is_partial = True
                    logger.error(
                        f"Segment {segment_id} attempt 3 validation also failed — marking partial"
                    )
            elif not statements:
                is_partial = True

        # Build archived_blocks payload (serialisable form, without raw_text at top level
        # to keep the extract compact — raw_text is included for traceability)
        archived_blocks = [
            {
                "block_id": b.block_id,
                "block_kind": b.block_kind,
                "speaker": b.speaker,
                "line_count": b.line_count,
                "char_count": b.char_count,
                "raw_text": b.raw_text,
                **({"archival_description": b.archival_description}
                   if b.archival_description else {}),
            }
            for b in blocks
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        output_data = self._build_output_document(
            statements=statements,
            segment=segment,
            archived_blocks=archived_blocks,
            extraction_attempt=extraction_attempt,
            latency_ms=latency_ms,
            is_degraded=is_degraded,
            is_partial=is_partial,
            skip_reason=skip_reason,
            warnings=warnings,
        )

        extract_dir.mkdir(parents=True, exist_ok=True)
        seg_idx = segment["segment_index"]
        json_path = extract_dir / f"extract_{seg_idx:03d}.json"
        md_path = extract_dir / f"extract_{seg_idx:03d}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_render_md(output_data))

        logger.debug(
            f"Segment {segment_id} extracted in {latency_ms}ms "
            f"({len(output_data.get('statements', []))} statements, "
            f"attempt={extraction_attempt})"
        )
        return extract_dir, output_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, segment_text: str) -> str:
        """Build a flat prompt string (system + user) for generate_json."""
        prompt_path = self._prompts_dir / "extract_segment.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()

        system_part, user_template = template.split("---USER---", 1)
        user_content = user_template.strip().replace("{segment_text}", segment_text)
        return system_part.strip() + "\n\n" + user_content

    def _describe_blocks(
        self, blocks: List[ArchivedBlock], segment_id: str
    ) -> List[ArchivedBlock]:
        """Run the archival-description LLM call for each block (attempt 2 only)."""
        prompt_path = self._prompts_dir / "describe_archived_block.md"
        if not prompt_path.exists():
            logger.warning(
                f"describe_archived_block.md prompt not found — skipping archival descriptions"
            )
            return blocks

        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()

        system_part, user_template = template.split("---USER---", 1)

        for blk in blocks:
            user_content = user_template.strip().replace("{block_text}", blk.raw_text)
            prompt = system_part.strip() + "\n\n" + user_content
            try:
                raw, _, _ = self._ollama.generate_json(prompt, _ARCHIVAL_SCHEMA)
                parsed, _, _ = self._ollama.parse_json_response(raw)
                if isinstance(parsed, dict):
                    blk.archival_description = parsed
            except (RuntimeError, ValueError) as e:
                logger.warning(
                    f"Segment {segment_id}: archival description failed for "
                    f"{blk.block_id}: {e}"
                )

        return blocks

    def _build_output_document(
        self,
        statements: List[Dict[str, Any]],
        segment: Dict[str, Any],
        archived_blocks: List[Dict[str, Any]],
        extraction_attempt: int,
        latency_ms: int,
        is_degraded: bool,
        is_partial: bool,
        skip_reason: Optional[str],
        warnings: List[str],
    ) -> Dict[str, Any]:
        # Filter malformed entries and inject per-statement metadata
        segment_id = segment["segment_id"]
        seg_idx = segment["segment_index"]
        conv_id = segment["conversation_id"]

        clean_statements = []
        dropped = 0
        for raw_s in statements:
            if not isinstance(raw_s, dict) or "speaker" not in raw_s or "text" not in raw_s:
                dropped += 1
                continue
            if raw_s["speaker"] not in ("user", "assistant"):
                dropped += 1
                continue
            clean_statements.append(raw_s)

        if dropped:
            logger.warning(
                f"Segment {segment_id}: dropped {dropped} malformed statement(s)"
            )

        # Inject deterministic per-statement metadata
        enriched = []
        for idx, s in enumerate(clean_statements):
            enriched.append({
                "statement_id": f"{segment_id}_st{idx:04d}",
                "statement_index": idx,
                "segment_id": segment_id,
                "segment_index": seg_idx,
                "parent_conversation_id": conv_id,
                "speaker": s["speaker"],
                "text": s["text"],
            })

        output: Dict[str, Any] = {
            "statements": enriched,
            "archived_blocks": archived_blocks,
            "extraction_attempt": extraction_attempt,
            "source_file": f"extract_{seg_idx:03d}.json",
            "source_kind": "ai_chat_extract",
            "segment_id": segment_id,
            "segment_index": seg_idx,
            "parent_conversation_id": conv_id,
            "conversation_date": segment.get("conversation_date"),
            "schema": self._schema,
            "schema_version": self._schema_version,
            "provider": "local",
            "model": self._ollama.model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "latency_ms": latency_ms,
        }

        if skip_reason:
            output["status"] = "skipped"
            output["warnings"] = [skip_reason] + [w for w in warnings if w]
        elif is_partial:
            output["status"] = "partial"
            output["warnings"] = [w for w in warnings if w]
        elif is_degraded:
            output["status"] = "degraded"
            output["warnings"] = [w for w in warnings if w]
        else:
            output["status"] = "ok"

        return output


# ---------------------------------------------------------------------------
# Validation / drift detection
# ---------------------------------------------------------------------------

def _validate_statements(
    parsed: Any,
    working_view: str,
) -> Tuple[bool, List[str]]:
    """Validate parsed model output; return (ok, warnings).

    Rejects on:
    - non-dict top-level or unexpected keys
    - invalid speaker values
    - implausibly high statement count
    - repetitive 4-gram drift
    - near-empty output on non-empty input
    """
    errs: List[str] = []

    if not isinstance(parsed, dict):
        return False, ["output is not a JSON object"]

    allowed_keys = {"statements"}
    unexpected = set(parsed.keys()) - allowed_keys
    if unexpected:
        errs.append(f"unexpected keys in output: {unexpected}")

    statements = parsed.get("statements")
    if not isinstance(statements, list):
        return False, errs + ["'statements' is not a list"]

    line_count = working_view.count("\n") + 1

    # Invalid speakers
    bad_speakers = [
        s.get("speaker") for s in statements
        if isinstance(s, dict) and s.get("speaker") not in ("user", "assistant")
    ]
    if bad_speakers:
        errs.append(f"invalid speaker value(s): {bad_speakers[:3]}")

    # Implausible count
    max_expected = max(40, line_count // 2)
    if len(statements) > max_expected:
        errs.append(
            f"statement count {len(statements)} exceeds plausible max {max_expected} "
            f"for {line_count}-line input"
        )

    # Near-empty output on non-empty input
    if len(working_view.strip()) > 200 and len(statements) == 0:
        errs.append("zero statements extracted from non-empty segment")

    # Drift detection: the model is "drifting" when it generates repetitive
    # near-duplicate statements. Structural repetition in the source (e.g. a
    # spec with bullet lists) produces shared phrases but distinct statements,
    # so we look at statement-level duplication, not raw n-gram counts.
    if statements:
        texts = [
            s["text"].strip().lower()
            for s in statements
            if isinstance(s, dict) and isinstance(s.get("text"), str)
        ]

        # (a) Near-identical statement duplicates — the strongest drift signal.
        # Count how many statements share a shingled signature with another.
        def _shingles(text: str, k: int = 5) -> frozenset:
            words = re.findall(r"\w+", text)
            if len(words) < k:
                return frozenset([" ".join(words)]) if words else frozenset()
            return frozenset(" ".join(words[i: i + k]) for i in range(len(words) - k + 1))

        sigs = [_shingles(t) for t in texts]
        dup_pairs = 0
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                a, b = sigs[i], sigs[j]
                if not a or not b:
                    continue
                inter = len(a & b)
                union = len(a | b)
                if union and inter / union >= 0.70:
                    dup_pairs += 1
        if dup_pairs >= 5:
            errs.append(
                f"found {dup_pairs} near-duplicate statement pair(s) — likely drift"
            )

        # (b) Dominant 4-gram — a 4-gram that appears in a large fraction of
        # statements (not just many times overall). Proportional threshold
        # avoids false positives on long extracts with legitimate repetition.
        if len(statements) >= 6:
            per_stmt_grams = []
            for t in texts:
                words = re.findall(r"\w+", t)
                grams = {tuple(words[i: i + 4]) for i in range(len(words) - 3)}
                per_stmt_grams.append(grams)

            gram_doc_freq: Dict[Tuple, int] = {}
            for grams in per_stmt_grams:
                for g in grams:
                    gram_doc_freq[g] = gram_doc_freq.get(g, 0) + 1

            n = len(per_stmt_grams)
            if gram_doc_freq:
                most_gram, most_count = max(gram_doc_freq.items(), key=lambda kv: kv[1])
                if most_count >= max(5, int(0.5 * n)):
                    errs.append(
                        f"4-gram {most_gram} appears in {most_count}/{n} statements "
                        f"— likely drift"
                    )

    if errs:
        return False, errs
    return True, []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_md(output_data: Dict[str, Any]) -> str:
    lines = [
        f"# Extract — {output_data['segment_id']}",
        "",
        f"**Status:** {output_data['status']}  ",
        f"**Attempt:** {output_data.get('extraction_attempt', '?')}  ",
        f"**Model:** {output_data['model']}  ",
        f"**Latency:** {output_data['latency_ms']}ms",
        "",
        "## Statements",
        "",
    ]
    for stmt in output_data.get("statements", []):
        lines.append(f"**{stmt.get('speaker', '?')}:** {stmt.get('text', '')}")
        lines.append("")

    archived = output_data.get("archived_blocks", [])
    if archived:
        lines += ["## Archived Blocks", ""]
        for blk in archived:
            lines.append(
                f"- `{blk['block_id']}` — {blk['block_kind']} "
                f"({blk['line_count']} lines, {blk['char_count']} chars, "
                f"speaker={blk.get('speaker') or 'unknown'})"
            )
            if blk.get("archival_description"):
                desc = blk["archival_description"]
                lines.append(f"  - {desc.get('brief_description', '')}")
        lines.append("")

    return "\n".join(lines)
