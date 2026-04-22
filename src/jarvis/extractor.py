"""Segment extractor for JARVIS.

Extracts all informational content from a conversation segment as a clean
list of attributed statements (speaker + text). This is the first step in
the extract→fragment retrieval pipeline.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jarvis.ollama import OllamaClient


logger = logging.getLogger(__name__)


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
                retries=retries,
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
        retries: int = 1,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Extract statements from a single segment.

        Args:
            segment: Segment dict (from segment_NNN.json).
            extract_dir: Directory to write extract artifacts.

        Returns:
            Tuple of (extract_dir, output_data).
        """
        start_time = time.time()

        system_prompt, user_content = self._build_prompt(segment["segment_text"])
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
                    f"Segment {segment['segment_id']} timed out on attempt {attempt + 1} — skipping"
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
                    f"Segment {segment['segment_id']} raw model output (attempt {attempt + 1}):\n{raw_response}"
                )
                if attempt < retries:
                    logger.warning(
                        f"Segment {segment['segment_id']} parse failed on attempt {attempt + 1} — retrying..."
                    )
                else:
                    logger.error(
                        f"Segment {segment['segment_id']} parse failed after {attempt + 1} attempt(s) — skipping"
                    )
                    is_partial = True
                    warning = f"JSON parse failed after {attempt + 1} attempt(s): {e}"

        latency_ms = int((time.time() - start_time) * 1000)

        if skip_reason:
            parsed_data = {"skipped": True}
            warning = skip_reason

        output_data = self._build_output_document(
            parsed_data=parsed_data,
            segment=segment,
            latency_ms=latency_ms,
            is_degraded=is_degraded,
            is_partial=is_partial,
            warning=warning,
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
            f"Segment {segment['segment_id']} extracted in {latency_ms}ms "
            f"({len(output_data.get('statements', []))} statements)"
        )
        return extract_dir, output_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, segment_text: str) -> tuple:
        prompt_path = self._prompts_dir / "extract_segment.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()

        system_prompt, user_template = template.split("---USER---", 1)
        user_content = user_template.strip().replace("{segment_text}", segment_text)
        return system_prompt.strip(), user_content

    def _build_output_document(
        self,
        parsed_data: Any,
        segment: Dict[str, Any],
        latency_ms: int,
        is_degraded: bool,
        is_partial: bool,
        warning: str,
    ) -> Dict[str, Any]:
        # Model sometimes returns a bare list instead of {"statements": [...]}
        if isinstance(parsed_data, list):
            raw_statements = parsed_data
        else:
            raw_statements = parsed_data.get("statements", [])
        # Filter out malformed entries missing required keys
        statements = [s for s in raw_statements if isinstance(s, dict) and "speaker" in s and "text" in s]
        if len(statements) < len(raw_statements):
            logger.warning(
                f"Segment {segment['segment_id']}: dropped {len(raw_statements) - len(statements)} "
                f"malformed statement(s) missing 'speaker' or 'text'"
            )
        seg_idx = segment["segment_index"]
        output: Dict[str, Any] = {
            "statements": statements,
            "source_file": f"extract_{seg_idx:03d}.json",
            "source_kind": "ai_chat_extract",
            "segment_id": segment["segment_id"],
            "segment_index": seg_idx,
            "parent_conversation_id": segment["conversation_id"],
            "conversation_date": segment.get("conversation_date"),
            "schema": self._schema,
            "schema_version": self._schema_version,
            "provider": "local",
            "model": self._ollama.model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "latency_ms": latency_ms,
        }

        if isinstance(parsed_data, dict) and parsed_data.get("skipped"):
            output["status"] = "skipped"
            output["warnings"] = [warning]
        elif is_partial:
            output["status"] = "partial"
            output["warnings"] = [warning]
        elif is_degraded:
            output["status"] = "degraded"
            output["warnings"] = [warning]
        else:
            output["status"] = "ok"

        return output


def _render_md(output_data: Dict[str, Any]) -> str:
    lines = [
        f"# Extract — {output_data['segment_id']}",
        "",
        f"**Status:** {output_data['status']}  ",
        f"**Model:** {output_data['model']}  ",
        f"**Latency:** {output_data['latency_ms']}ms",
        "",
        "## Statements",
        "",
    ]
    for stmt in output_data.get("statements", []):
        lines.append(f"**{stmt.get('speaker', '?')}:** {stmt.get('text', '')}")
        lines.append("")
    return "\n".join(lines)
