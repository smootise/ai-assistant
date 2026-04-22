"""Segment summarizer for AI chat conversations.

Summarizes each segment produced by the segmenter, passing the last N prior
segment summaries as rolling context so the model understands continuity
without re-summarizing earlier topics.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jarvis.ollama import OllamaClient
from jarvis.output_writer import OutputWriter


logger = logging.getLogger(__name__)


class SegmentSummarizer:
    """Summarizes conversation segments with rolling context."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        prompts_dir: str,
        schema: str,
        schema_version: str,
        context_window: int = 3,
    ):
        self.ollama = ollama_client
        self.prompts_dir = Path(prompts_dir)
        self.schema = schema
        self.schema_version = schema_version
        self.context_window = context_window

    def summarize_conversation_segments(
        self,
        segments_dir: Path,
        conversation_id: str,
        output_root: Path,
        from_segment: int = 0,
        to_segment: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> List[Tuple[Path, Dict[str, Any]]]:
        """Summarize segments in [from_segment, to_segment] with rolling context.

        On re-runs, pre-seeds the rolling context from any already-written
        summary files for segments before from_segment, so partial runs stay
        contextually correct.

        Args:
            segments_dir: Directory containing segment_NNN.json files.
            conversation_id: Parent conversation ID (used for output paths).
            output_root: Root output directory (e.g. OUTPUTS/).
            from_segment: First segment index to summarize (inclusive).
            to_segment: Last segment index to summarize (inclusive). None = last.
            run_id: Optional run identifier propagated to each summary.

        Returns:
            List of (output_dir, output_data) tuples, one per summarized segment.

        Raises:
            FileNotFoundError: If segments_dir does not exist.
        """
        if not segments_dir.exists():
            raise FileNotFoundError(f"Segments directory not found: {segments_dir}")

        segment_summaries_dir = output_root / conversation_id / "segment_summaries"

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
                f"No segments in range [{from_segment}, {effective_to}] — nothing to summarize."
            )
            return []

        rolling_summaries: List[str] = self._load_existing_summaries(
            segment_summaries_dir, segments, before_index=from_segment
        )
        logger.info(
            f"Pre-seeded {len(rolling_summaries)} prior summaries for context "
            f"(from_segment={from_segment})"
        )

        results: List[Tuple[Path, Dict[str, Any]]] = []
        for segment in selected:
            existing_path = segment_summaries_dir / f"{segment['segment_id']}.json"
            if existing_path.exists():
                logger.info(
                    f"Skipping segment {segment['segment_index']} / "
                    f"{segments[-1]['segment_index']} "
                    f"({segment['segment_id']}) — summary already exists"
                )
                with open(existing_path, encoding="utf-8") as f:
                    output_data = json.load(f)
                rolling_summaries.append(output_data.get("summary", ""))
                results.append((segment_summaries_dir, output_data))
                continue

            logger.info(
                f"Summarizing segment {segment['segment_index']} / "
                f"{segments[-1]['segment_index']} ({segment['segment_id']})"
            )
            prior = rolling_summaries[-self.context_window:]
            output_dir, output_data = self.summarize_segment(
                segment=segment,
                prior_summaries=prior,
                segment_summaries_dir=segment_summaries_dir,
                run_id=run_id,
            )
            rolling_summaries.append(output_data["summary"])
            results.append((output_dir, output_data))

        logger.info(f"Summarized {len(results)} segments for conversation {conversation_id}")
        return results

    def summarize_segment(
        self,
        segment: Dict[str, Any],
        prior_summaries: List[str],
        segment_summaries_dir: Path,
        run_id: Optional[str] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Summarize a single segment dict.

        Args:
            segment: Segment dict (from segment_NNN.json).
            prior_summaries: Ordered list of prior segment summary strings (context).
            segment_summaries_dir: Directory to write the summary artifacts.
            run_id: Optional run identifier.

        Returns:
            Tuple of (output_dir, output_data).
        """
        start_time = time.time()

        system_prompt, user_content = self._build_segment_prompt(segment["segment_text"], prior_summaries)
        raw_response, is_degraded, warning = self.ollama.chat(system_prompt, user_content)

        try:
            parsed_data, parse_degraded, parse_warning = self.ollama.parse_json_response(
                raw_response
            )
            if parse_degraded:
                is_degraded = True
                warning = parse_warning
        except ValueError as e:
            logger.error(f"Failed to parse response for segment {segment['segment_id']}: {e}")
            raise RuntimeError(f"Model did not return valid JSON: {e}") from e

        latency_ms = int((time.time() - start_time) * 1000)

        output_data = self._build_output_document(
            parsed_data=parsed_data,
            segment=segment,
            latency_ms=latency_ms,
            is_degraded=is_degraded,
            warning=warning,
            run_id=run_id,
        )

        writer = OutputWriter(output_root=str(segment_summaries_dir), use_timestamp=False)
        output_dir = writer.write_outputs(
            summary_data=output_data,
            source_file=f"{segment['segment_id']}.json",
            run_id=run_id,
        )

        logger.debug(f"Segment {segment['segment_id']} summarized in {latency_ms}ms")
        return output_dir, output_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_segment_prompt(self, segment_text: str, prior_summaries: List[str]) -> tuple:
        prompt_path = self.prompts_dir / "summarize_ai_chat_segment.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, encoding="utf-8") as f:
            system_prompt = f.read().strip()

        user_parts = []
        if prior_summaries:
            context_lines = "\n".join(
                f"Segment {i}: {s}" for i, s in enumerate(prior_summaries)
            )
            user_parts.append(
                f"---BEGIN PREVIOUS CONTEXT---\n{context_lines}\n---END PREVIOUS CONTEXT---"
            )

        user_parts.append(
            f"---BEGIN SEGMENT TRANSCRIPT---\n{segment_text}\n---END SEGMENT TRANSCRIPT---"
        )

        return system_prompt, "\n\n".join(user_parts)

    def _build_output_document(
        self,
        parsed_data: Dict[str, Any],
        segment: Dict[str, Any],
        latency_ms: int,
        is_degraded: bool,
        warning: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "summary": parsed_data.get("summary", ""),
            "bullets": parsed_data.get("bullets", []),
            "action_items": parsed_data.get("action_items", []),
            "confidence": parsed_data.get("confidence", 0.0),
            "schema": self.schema,
            "schema_version": self.schema_version,
            "provider": "local",
            "model": self.ollama.model,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_file": f"{segment['segment_id']}.json",
            "source_kind": "ai_chat_segment",
            "latency_ms": latency_ms,
            "segment_id": segment["segment_id"],
            "segment_index": segment["segment_index"],
            "parent_conversation_id": segment["conversation_id"],
        }

        if is_degraded:
            output["status"] = "degraded"
            output["warnings"] = [warning]
        else:
            output["status"] = "ok"

        if run_id:
            output["run_id"] = run_id

        if "lang" in parsed_data:
            output["lang"] = parsed_data["lang"]

        return output

    def _load_existing_summaries(
        self,
        segment_summaries_dir: Path,
        all_segments: List[Dict[str, Any]],
        before_index: int,
    ) -> List[str]:
        if before_index == 0 or not segment_summaries_dir.exists():
            return []

        summaries = []
        for segment in all_segments:
            if segment["segment_index"] >= before_index:
                break
            summary_path = segment_summaries_dir / f"{segment['segment_id']}.json"
            if summary_path.exists():
                try:
                    with open(summary_path, encoding="utf-8") as f:
                        data = json.load(f)
                    summaries.append(data.get("summary", ""))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Could not load existing summary {summary_path}: {e}")
        return summaries
