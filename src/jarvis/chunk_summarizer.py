"""Chunk summarizer for AI chat conversation segments.

Summarizes each chunk produced by the chunker, passing the last N prior
chunk summaries as rolling context so the model understands continuity
without re-summarizing earlier segments.
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


class ChunkSummarizer:
    """Summarizes conversation chunks with rolling context."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        prompts_dir: str,
        schema: str,
        schema_version: str,
        context_window: int = 3,
    ):
        """Initialize the chunk summarizer.

        Args:
            ollama_client: Ollama API client.
            prompts_dir: Directory containing prompt templates.
            schema: Schema identifier (e.g., 'jarvis.summarization').
            schema_version: Schema version (e.g., '1.0.0').
            context_window: Number of prior chunk summaries to include as context.
        """
        self.ollama = ollama_client
        self.prompts_dir = Path(prompts_dir)
        self.schema = schema
        self.schema_version = schema_version
        self.context_window = context_window

    def summarize_conversation_chunks(
        self,
        chunks_dir: Path,
        conversation_id: str,
        output_root: Path,
        from_chunk: int = 0,
        to_chunk: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> List[Tuple[Path, Dict[str, Any]]]:
        """Summarize chunks in [from_chunk, to_chunk] with rolling context.

        On re-runs, pre-seeds the rolling context from any already-written
        summary files for chunks before from_chunk, so partial runs stay
        contextually correct.

        Args:
            chunks_dir: Directory containing chunk_NNN.json files.
            conversation_id: Parent conversation ID (used for output paths).
            output_root: Root output directory (e.g. OUTPUTS/).
            from_chunk: First chunk index to summarize (inclusive).
            to_chunk: Last chunk index to summarize (inclusive). None = last.
            run_id: Optional run identifier propagated to each summary.

        Returns:
            List of (output_dir, output_data) tuples, one per summarized chunk.

        Raises:
            FileNotFoundError: If chunks_dir does not exist.
        """
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

        chunk_summaries_dir = output_root / conversation_id / "chunk_summaries"

        # Load all chunk files, sorted by index, excluding pending_tail
        chunk_files = sorted(
            f for f in chunks_dir.glob("chunk_*.json") if f.name != "pending_tail.json"
        )

        chunks = []
        for path in chunk_files:
            with open(path, encoding="utf-8") as f:
                chunks.append(json.load(f))

        # Sort by chunk_index to be safe
        chunks.sort(key=lambda c: c["chunk_index"])

        # Apply range filter
        effective_to = to_chunk if to_chunk is not None else chunks[-1]["chunk_index"]
        selected = [c for c in chunks if from_chunk <= c["chunk_index"] <= effective_to]

        if not selected:
            logger.warning(
                f"No chunks in range [{from_chunk}, {effective_to}] — nothing to summarize."
            )
            return []

        # Pre-seed rolling context from existing summaries before from_chunk
        rolling_summaries: List[str] = self._load_existing_summaries(
            chunk_summaries_dir, chunks, before_index=from_chunk
        )
        logger.info(
            f"Pre-seeded {len(rolling_summaries)} prior summaries for context "
            f"(from_chunk={from_chunk})"
        )

        results: List[Tuple[Path, Dict[str, Any]]] = []
        for chunk in selected:
            logger.info(
                f"Summarizing chunk {chunk['chunk_index']} / {chunks[-1]['chunk_index']} "
                f"({chunk['chunk_id']})"
            )
            prior = rolling_summaries[-self.context_window:]
            output_dir, output_data = self.summarize_chunk(
                chunk=chunk,
                prior_summaries=prior,
                chunk_summaries_dir=chunk_summaries_dir,
                run_id=run_id,
            )
            rolling_summaries.append(output_data["summary"])
            results.append((output_dir, output_data))

        logger.info(f"Summarized {len(results)} chunks for conversation {conversation_id}")
        return results

    def summarize_chunk(
        self,
        chunk: Dict[str, Any],
        prior_summaries: List[str],
        chunk_summaries_dir: Path,
        run_id: Optional[str] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Summarize a single chunk dict.

        Args:
            chunk: Chunk dict (from chunk_NNN.json).
            prior_summaries: Ordered list of prior chunk summary strings (context).
            chunk_summaries_dir: Directory to write the summary artifacts.
            run_id: Optional run identifier.

        Returns:
            Tuple of (output_dir, output_data).
        """
        start_time = time.time()

        prompt = self._build_chunk_prompt(chunk["chunk_text"], prior_summaries)
        raw_response, is_degraded, warning = self.ollama.generate(prompt)

        try:
            parsed_data, parse_degraded, parse_warning = self.ollama.parse_json_response(
                raw_response
            )
            if parse_degraded:
                is_degraded = True
                warning = parse_warning
        except ValueError as e:
            logger.error(f"Failed to parse response for chunk {chunk['chunk_id']}: {e}")
            raise RuntimeError(f"Model did not return valid JSON: {e}") from e

        latency_ms = int((time.time() - start_time) * 1000)

        output_data = self._build_output_document(
            parsed_data=parsed_data,
            chunk=chunk,
            latency_ms=latency_ms,
            is_degraded=is_degraded,
            warning=warning,
            run_id=run_id,
        )

        writer = OutputWriter(output_root=str(chunk_summaries_dir), use_timestamp=False)
        output_dir = writer.write_outputs(
            summary_data=output_data,
            source_file=f"{chunk['chunk_id']}.json",
            run_id=run_id,
        )

        logger.debug(f"Chunk {chunk['chunk_id']} summarized in {latency_ms}ms")
        return output_dir, output_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_chunk_prompt(self, chunk_text: str, prior_summaries: List[str]) -> str:
        """Build the full prompt for a single chunk.

        Args:
            chunk_text: Formatted transcript text for this chunk.
            prior_summaries: Ordered list of prior chunk summary strings.

        Returns:
            Complete prompt string.

        Raises:
            FileNotFoundError: If the prompt template is missing.
        """
        prompt_path = self.prompts_dir / "summarize_ai_chat_chunk.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()

        parts = [template]

        if prior_summaries:
            context_lines = "\n".join(
                f"Chunk {i}: {s}" for i, s in enumerate(prior_summaries)
            )
            parts.append(
                f"\n\n---BEGIN PREVIOUS CONTEXT---\n{context_lines}\n---END PREVIOUS CONTEXT---"
            )

        parts.append(f"\n\n---BEGIN CHUNK TRANSCRIPT---\n{chunk_text}\n---END CHUNK TRANSCRIPT---")

        return "".join(parts)

    def _build_output_document(
        self,
        parsed_data: Dict[str, Any],
        chunk: Dict[str, Any],
        latency_ms: int,
        is_degraded: bool,
        warning: str,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the complete output document for a chunk summary.

        Args:
            parsed_data: Parsed model output (summary, bullets, etc.).
            chunk: The source chunk dict.
            latency_ms: Inference latency in milliseconds.
            is_degraded: Whether output is degraded.
            warning: Warning message if degraded.
            run_id: Optional run identifier.

        Returns:
            Complete output dictionary.
        """
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
            "source_file": f"{chunk['chunk_id']}.json",
            "source_kind": "ai_chat_chunk",
            "latency_ms": latency_ms,
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "parent_conversation_id": chunk["conversation_id"],
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
        chunk_summaries_dir: Path,
        all_chunks: List[Dict[str, Any]],
        before_index: int,
    ) -> List[str]:
        """Load summary strings from already-written files for chunks before before_index.

        Used to pre-seed the rolling context on re-runs or partial runs.

        Args:
            chunk_summaries_dir: Directory containing <chunk_id>.json summary files.
            all_chunks: All chunks for the conversation (sorted by chunk_index).
            before_index: Only load summaries for chunks with index < before_index.

        Returns:
            Ordered list of summary strings.
        """
        if before_index == 0 or not chunk_summaries_dir.exists():
            return []

        summaries = []
        for chunk in all_chunks:
            if chunk["chunk_index"] >= before_index:
                break
            summary_path = chunk_summaries_dir / f"{chunk['chunk_id']}.json"
            if summary_path.exists():
                try:
                    with open(summary_path, encoding="utf-8") as f:
                        data = json.load(f)
                    summaries.append(data.get("summary", ""))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Could not load existing summary {summary_path}: {e}")
        return summaries
