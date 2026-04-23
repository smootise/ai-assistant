"""Fragment generator for JARVIS.

Takes the extracted statements from a segment and groups them into topically
coherent fragments. Each fragment is an independent retrieval unit.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jarvis.ollama import OllamaClient


logger = logging.getLogger(__name__)


class Fragmenter:
    """Groups extracted statements into topically coherent retrieval fragments."""

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

    def fragment_conversation_extracts(
        self,
        conversation_id: str,
        output_root: Path,
        from_segment: int = 0,
        to_segment: Optional[int] = None,
        force: bool = False,
        retries: int = 1,
    ) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[Tuple[str, str]]]:
        """Fragment extracts for a conversation, optionally within a segment range.

        Resume-safe: skips segments whose fragment files already exist unless
        force=True (detected by presence of fragment_000.json).

        Args:
            conversation_id: Parent conversation ID.
            output_root: Root output directory (e.g. OUTPUTS/).
            from_segment: First segment index to process (inclusive).
            to_segment: Last segment index to process (inclusive). None = last.
            force: If True, overwrite existing fragment files.

        Returns:
            Tuple of:
              - List of (fragment_dir, output_data) tuples, one per fragment.
              - List of (segment_id, reason) tuples for segments that were skipped.

        Raises:
            FileNotFoundError: If the extracts directory does not exist.
        """
        extract_dir = output_root / conversation_id / "extracts"
        if not extract_dir.exists():
            raise FileNotFoundError(f"Extracts directory not found: {extract_dir}")

        fragments_root = output_root / conversation_id / "fragments"

        extract_files = sorted(extract_dir.glob("extract_*.json"))
        if not extract_files:
            logger.warning(f"No extract files found in {extract_dir}")
            return [], []

        results: List[Tuple[Path, Dict[str, Any]]] = []
        skipped: List[Tuple[str, str]] = []
        for extract_path in extract_files:
            with open(extract_path, encoding="utf-8") as f:
                extract_data = json.load(f)

            seg_idx = extract_data["segment_index"]
            effective_to = to_segment if to_segment is not None else float("inf")
            if not (from_segment <= seg_idx <= effective_to):
                continue

            # Skip extracts that were themselves skipped (timeout/parse failure)
            if extract_data.get("status") == "skipped":
                reason = (extract_data.get("warnings") or ["skipped during extraction"])[0]
                skipped.append((extract_data["segment_id"], f"extract skipped: {reason}"))
                continue

            segment_fragment_dir = fragments_root / f"segment_{seg_idx:03d}"
            sentinel = segment_fragment_dir / "fragment_000.json"

            if sentinel.exists() and not force:
                with open(sentinel, encoding="utf-8") as f:
                    sentinel_data = json.load(f)
                if sentinel_data.get("status") in ("skipped", "partial"):
                    logger.info(
                        f"Re-processing segment {seg_idx} ({extract_data['segment_id']}) "
                        f"— previous fragments have status '{sentinel_data['status']}'"
                    )
                    for frag_file in segment_fragment_dir.iterdir():
                        frag_file.unlink()
                else:
                    logger.info(
                        f"Skipping segment {seg_idx} ({extract_data['segment_id']}) "
                        f"— fragments already exist"
                    )
                    for frag_path in sorted(segment_fragment_dir.glob("fragment_*.json")):
                        with open(frag_path, encoding="utf-8") as f:
                            results.append((segment_fragment_dir, json.load(f)))
                    continue

            if force and segment_fragment_dir.exists():
                for f in segment_fragment_dir.iterdir():
                    f.unlink()
                logger.info(f"--force: cleared {segment_fragment_dir}")

            logger.info(
                f"Fragmenting segment {seg_idx} ({extract_data['segment_id']})"
            )
            fragments = self.fragment_extract(
                extract_data=extract_data,
                fragment_dir=segment_fragment_dir,
                retries=retries,
            )
            if not fragments:
                skipped.append((extract_data["segment_id"], "fragmentation produced no output (timeout or parse failure)"))
            results.extend(fragments)

        logger.info(
            f"Produced {len(results)} fragments for conversation {conversation_id}"
            + (f" ({len(skipped)} segments skipped)" if skipped else "")
        )
        return results, skipped

    def fragment_extract(
        self,
        extract_data: Dict[str, Any],
        fragment_dir: Path,
        retries: int = 1,
    ) -> List[Tuple[Path, Dict[str, Any]]]:
        """Fragment a single extract into topically coherent sub-units.

        Args:
            extract_data: Extract output_data dict (from extract_NNN.json).
            fragment_dir: Directory to write fragment artifacts.

        Returns:
            List of (fragment_dir, output_data) tuples, one per fragment.
        """
        statements = extract_data.get("statements", [])
        if not statements:
            logger.warning(
                f"No statements in extract for segment {extract_data.get('segment_id')} "
                f"— skipping fragmentation"
            )
            return []

        start_time = time.time()

        system_prompt, user_content = self._build_prompt(statements)
        parsed_data: Dict[str, Any] = {}
        is_degraded = False
        is_partial = False
        warning = ""

        skip_reason: Optional[str] = None
        for attempt in range(retries + 1):
            try:
                raw_response, gen_degraded, gen_warning = self._ollama.chat(system_prompt, user_content)
            except RuntimeError as e:
                skip_reason = f"Timeout on attempt {attempt + 1}: {e}"
                logger.error(
                    f"Segment {extract_data.get('segment_id')} fragment timed out "
                    f"on attempt {attempt + 1} — skipping"
                )
                break
            try:
                parsed_data, parse_degraded, parse_warning = self._ollama.parse_json_response(
                    raw_response
                )
                is_degraded = gen_degraded or parse_degraded
                warning = parse_warning if parse_degraded else gen_warning
                break
            except ValueError as e:
                logger.debug(
                    f"Segment {extract_data.get('segment_id')} raw model output (attempt {attempt + 1}):\n{raw_response}"
                )
                if attempt < retries:
                    logger.warning(
                        f"Segment {extract_data.get('segment_id')} fragment parse failed "
                        f"on attempt {attempt + 1} — retrying..."
                    )
                else:
                    logger.error(
                        f"Segment {extract_data.get('segment_id')} fragment parse failed "
                        f"after {attempt + 1} attempt(s) — skipping"
                    )
                    is_partial = True
                    warning = f"JSON parse failed after {attempt + 1} attempt(s): {e}"

        latency_ms = int((time.time() - start_time) * 1000)

        if skip_reason:
            logger.warning(
                f"Segment {extract_data.get('segment_id')} fragment skipped: {skip_reason}"
            )
            return []

        # Model sometimes returns a bare list instead of {"fragments": [...]}
        if isinstance(parsed_data, list):
            raw_fragments = parsed_data
        else:
            raw_fragments = parsed_data.get("fragments", [])
        if not raw_fragments:
            logger.warning(
                f"Model returned no fragments for segment {extract_data.get('segment_id')}"
            )
            return []

        fragment_dir.mkdir(parents=True, exist_ok=True)
        seg_idx = extract_data["segment_index"]

        results: List[Tuple[Path, Dict[str, Any]]] = []
        for frag_idx, raw_frag in enumerate(raw_fragments):
            output_data = self._build_output_document(
                raw_fragment=raw_frag,
                extract_data=extract_data,
                seg_idx=seg_idx,
                frag_idx=frag_idx,
                latency_ms=latency_ms,
                is_degraded=is_degraded,
                is_partial=is_partial,
                warning=warning,
            )

            json_path = fragment_dir / f"fragment_{frag_idx:03d}.json"
            md_path = fragment_dir / f"fragment_{frag_idx:03d}.md"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(_render_md(output_data))

            results.append((fragment_dir, output_data))

        logger.debug(
            f"Segment {extract_data['segment_id']} fragmented into "
            f"{len(results)} fragments in {latency_ms}ms"
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, statements: List[Dict[str, Any]]) -> tuple:
        prompt_path = self._prompts_dir / "fragment_extract.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()

        statements_text = "\n".join(
            f"{stmt['speaker']}: {stmt['text']}" for stmt in statements
        )
        system_prompt, user_template = template.split("---USER---", 1)
        user_content = user_template.strip().replace("{statements_text}", statements_text)
        return system_prompt.strip(), user_content

    def _build_output_document(
        self,
        raw_fragment: Dict[str, Any],
        extract_data: Dict[str, Any],
        seg_idx: int,
        frag_idx: int,
        latency_ms: int,
        is_degraded: bool,
        is_partial: bool,
        warning: str,
    ) -> Dict[str, Any]:
        frag_statements = raw_fragment.get("statements", [])
        title = raw_fragment.get("title", "")

        # Build a lookup from [ARCHIVED_BLOCK_N] token → brief_description.
        # Keys include brackets to match what re.sub group(0) returns.
        archived_lookup: Dict[str, str] = {}
        for blk in extract_data.get("archived_blocks", []):
            bid = blk.get("block_id", "")           # e.g. "ARCHIVED_BLOCK_1"
            token = f"[{bid}]"                       # e.g. "[ARCHIVED_BLOCK_1]"
            desc = blk.get("archival_description") or {}
            brief = desc.get("brief_description") or (
                f"{blk.get('block_kind', 'block')} "
                f"({blk.get('line_count', '?')} lines)"
            )
            archived_lookup[token] = brief

        # Build retrieval-safe text: replace any [ARCHIVED_BLOCK_N] tokens with
        # their archival description; never include raw block content.
        def _clean_stmt_text(text: str) -> str:
            def _replace(m: re.Match) -> str:
                return archived_lookup.get(m.group(0), m.group(0))
            return re.sub(r"\[ARCHIVED_BLOCK_\d+\]", _replace, text)

        cleaned_lines = [
            f"{s.get('speaker', '?')}: {_clean_stmt_text(s.get('text', ''))}"
            for s in frag_statements
        ]

        # Collect any archival block descriptions referenced by this fragment's statements
        referenced_tokens = set()
        for s in frag_statements:
            for m in re.finditer(r"\[ARCHIVED_BLOCK_\d+\]", s.get("text", "")):
                referenced_tokens.add(m.group(0))

        archival_notes = [
            f"[Archived: {archived_lookup[token]}]"
            for token in sorted(referenced_tokens)
            if token in archived_lookup
        ]

        text_parts = []
        if title:
            text_parts.append(title)
        if cleaned_lines:
            text_parts.append("\n".join(cleaned_lines))
        if archival_notes:
            text_parts.append("\n".join(archival_notes))
        text = "\n\n".join(text_parts)

        # Compute statement index span for traceability back to the extract
        stmt_indices = [
            s.get("statement_index")
            for s in frag_statements
            if isinstance(s.get("statement_index"), int)
        ]
        statement_start_index = min(stmt_indices) if stmt_indices else None
        statement_end_index = max(stmt_indices) if stmt_indices else None

        output: Dict[str, Any] = {
            "text": text,
            "title": title,
            "statements": frag_statements,
            "statement_start_index": statement_start_index,
            "statement_end_index": statement_end_index,
            "source_file": f"segment_{seg_idx:03d}/fragment_{frag_idx:03d}.json",
            "source_kind": "ai_chat_fragment",
            "segment_id": extract_data["segment_id"],
            "segment_index": seg_idx,
            "fragment_index": frag_idx,
            "parent_conversation_id": extract_data["parent_conversation_id"],
            "conversation_date": extract_data.get("conversation_date"),
            "schema": self._schema,
            "schema_version": self._schema_version,
            "provider": "local",
            "model": self._ollama.model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "latency_ms": latency_ms,
        }

        if is_partial:
            output["status"] = "partial"
            output["warnings"] = [warning]
        elif is_degraded:
            output["status"] = "degraded"
            output["warnings"] = [warning]
        else:
            output["status"] = "ok"

        return output


def _render_md(output_data: Dict[str, Any]) -> str:
    title = output_data.get("title", "")
    lines = [
        f"# Fragment — {output_data['segment_id']} / {output_data['fragment_index']}",
        "",
        f"**Title:** {title}  ",
        f"**Status:** {output_data['status']}  ",
        f"**Model:** {output_data['model']}",
        "",
        "## Statements",
        "",
    ]
    for stmt in output_data.get("statements", []):
        lines.append(f"**{stmt.get('speaker', '?')}:** {stmt.get('text', '')}")
        lines.append("")
    return "\n".join(lines)
