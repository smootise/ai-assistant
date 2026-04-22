"""Topic detection and summarization for JARVIS.

Groups consecutive segment summaries into topics by measuring cosine
similarity between adjacent segment summary embeddings. When similarity drops
below the threshold a new topic starts. Each topic is then summarized
with a single LLM call.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jarvis.embedder import EmbeddingClient
from jarvis.ollama import OllamaClient
from jarvis.output_writer import OutputWriter
from jarvis.vector_store import VectorStore


logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python, no numpy)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _build_embedding_text(segment_summary: Dict[str, Any]) -> str:
    """Build the text to embed for topic boundary detection.

    Uses summary text only (not bullets/action_items) so consecutive segments on
    the same topic score high and genuine topic shifts score low. The richer
    retrieval embedding stored in Qdrant is not suitable here because its
    multi-field construction flattens inter-segment similarity across the board.
    """
    return segment_summary.get("summary", "")


class TopicDetector:
    """Detects topic boundaries across segments and summarizes each topic."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        ollama_client: OllamaClient,
        prompts_dir: str,
        schema: str,
        schema_version: str,
        threshold: float = 0.55,
        vector_store: Optional[VectorStore] = None,
    ):
        self._embedder = embedder
        self._ollama = ollama_client
        self._prompts_dir = Path(prompts_dir)
        self._schema = schema
        self._schema_version = schema_version
        self._threshold = threshold
        self._vector_store = vector_store
        self._prompt_template: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_summarize(
        self,
        conversation_id: str,
        output_root: Path,
        dry_run: bool = False,
        run_id: Optional[str] = None,
    ) -> List[Tuple[Path, Dict[str, Any]]]:
        """Full pipeline: load embeddings → detect boundaries → summarize topics.

        Args:
            conversation_id: ID of the conversation to process.
            output_root: Root directory where OUTPUTS are stored.
            dry_run: If True, detect boundaries but skip LLM summarization.
            run_id: Optional run identifier to propagate to output documents.

        Returns:
            List of (output_dir, output_data) tuples, one per topic.
            Empty list when dry_run=True.

        Raises:
            FileNotFoundError: If no segment summary files are found.
        """
        segment_summaries_dir = output_root / conversation_id / "segment_summaries"
        if not segment_summaries_dir.exists():
            raise FileNotFoundError(
                f"Segment summaries directory not found: {segment_summaries_dir}. "
                "Run 'summarize-segments' first."
            )

        segment_files = sorted(segment_summaries_dir.glob("*.json"))
        if not segment_files:
            raise FileNotFoundError(
                f"No segment summary JSON files found in {segment_summaries_dir}."
            )

        segment_summaries = []
        for f in segment_files:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("source_kind") == "ai_chat_segment":
                segment_summaries.append(data)

        if not segment_summaries:
            raise FileNotFoundError(
                f"No ai_chat_segment summaries found in {segment_summaries_dir}."
            )

        segment_summaries.sort(key=lambda d: d.get("segment_index", 0))

        topic_summaries_dir = output_root / conversation_id / "topic_summaries"
        if topic_summaries_dir.exists():
            existing = list(topic_summaries_dir.glob("topic_*.json"))
            if existing:
                logger.warning(
                    f"{len(existing)} existing topic files found — old files with higher "
                    "indices may become stale if topic count changes."
                )

        vectors = self._load_embeddings(conversation_id, segment_summaries)
        topics, similarities = self.detect_topics(segment_summaries, vectors)

        self._print_distribution_report(similarities, topics, segment_summaries)

        if dry_run:
            self._print_dry_run_plan(conversation_id, topics)
            return []

        results = []
        for topic_idx, topic_segments in enumerate(topics):
            output_dir, output_data = self.summarize_topic(
                topic_segments=topic_segments,
                topic_index=topic_idx,
                conversation_id=conversation_id,
                topic_summaries_dir=topic_summaries_dir,
                run_id=run_id,
            )
            results.append((output_dir, output_data))

        return results

    def detect_topics(
        self,
        segment_summaries: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> Tuple[List[List[Dict[str, Any]]], List[float]]:
        """Group segments into topics based on cosine similarity.

        Args:
            segment_summaries: Ordered list of segment summary dicts.
            vectors: Corresponding embedding vectors (same order).

        Returns:
            Tuple of (topics, similarities) where topics is a list of segment
            groups and similarities is the list of consecutive pair similarities.
        """
        if not segment_summaries:
            return [], []

        if len(segment_summaries) == 1:
            return [segment_summaries], []

        similarities: List[float] = []
        topics: List[List[Dict[str, Any]]] = [[segment_summaries[0]]]
        topic_num = 0

        for i in range(1, len(segment_summaries)):
            sim = _cosine_similarity(vectors[i - 1], vectors[i])
            similarities.append(sim)

            if sim < self._threshold:
                seg_id = segment_summaries[i - 1].get("segment_id", f"segment_{i - 1:03d}")
                logger.info(
                    f"Boundary after {seg_id} (similarity={sim:.3f} < "
                    f"threshold={self._threshold}) → topic {topic_num + 1}"
                )
                topic_num += 1
                topics.append([])

            topics[-1].append(segment_summaries[i])

        return topics, similarities

    def summarize_topic(
        self,
        topic_segments: List[Dict[str, Any]],
        topic_index: int,
        conversation_id: str,
        topic_summaries_dir: Path,
        run_id: Optional[str] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Summarize one topic with the LLM.

        Args:
            topic_segments: Segment summary dicts that form this topic.
            topic_index: Zero-based index of this topic.
            conversation_id: Parent conversation ID.
            topic_summaries_dir: Directory to write output files into.
            run_id: Optional run identifier.

        Returns:
            Tuple of (output_dir, output_data).
        """
        start_time = time.time()
        system_prompt, user_content = self._build_topic_prompt(topic_segments)
        raw, is_degraded, warning = self._ollama.chat(system_prompt, user_content)
        try:
            parsed, parse_degraded, parse_warning = self._ollama.parse_json_response(raw)
            if parse_degraded:
                is_degraded = True
                warning = parse_warning
        except ValueError as e:
            logger.error(f"Failed to parse response for topic {topic_index}: {e}")
            raise RuntimeError(f"Model did not return valid JSON: {e}") from e
        latency_ms = int((time.time() - start_time) * 1000)

        first_seg_id = topic_segments[0].get("segment_id", "s000")
        last_seg_id = topic_segments[-1].get("segment_id", f"s{len(topic_segments) - 1:03d}")
        segment_range = f"{first_seg_id}-{last_seg_id}"

        output_data = {
            **parsed,
            "source_file": f"topic_{topic_index:03d}.json",
            "source_kind": "ai_chat_topic",
            "topic_index": topic_index,
            "topic_segment_range": segment_range,
            "parent_conversation_id": conversation_id,
            "provider": "local",
            "model": self._ollama.model,
            "schema": self._schema,
            "schema_version": self._schema_version,
            "latency_ms": latency_ms,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "ok",
        }
        if is_degraded:
            output_data["status"] = "degraded"
            output_data.setdefault("warnings", [])
            if warning:
                output_data["warnings"].append(warning)
        if run_id:
            output_data["run_id"] = run_id

        writer = OutputWriter(
            output_root=str(topic_summaries_dir),
            use_timestamp=False,
        )
        output_dir = writer.write_outputs(
            summary_data=output_data,
            source_file=f"topic_{topic_index:03d}.json",
        )
        logger.info(
            f"Topic {topic_index} summarized → {output_dir} "
            f"(segments {segment_range})"
        )
        return output_dir, output_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_embeddings(
        self,
        conversation_id: str,
        segment_summaries: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """Embed each segment summary for boundary detection.

        Always embeds on the fly from summary text only. Qdrant vectors are
        intentionally not reused here — they are built from a multi-field
        retrieval payload that suppresses inter-segment similarity and produces
        a flat distribution unsuitable for boundary detection.

        Args:
            conversation_id: Unused; kept for API consistency.
            segment_summaries: Segment summary dicts to embed.

        Returns:
            List of float vectors in the same order as segment_summaries.
        """
        logger.info(
            f"Embedding {len(segment_summaries)} segment summaries for topic detection"
        )
        return [
            self._embedder.embed(_build_embedding_text(ss))
            for ss in segment_summaries
        ]

    def _load_prompt_template(self) -> str:
        """Load the topic summarization prompt template (cached)."""
        if self._prompt_template is None:
            prompt_path = self._prompts_dir / "summarize_topic.md"
            self._prompt_template = prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    def _build_topic_prompt(self, topic_segments: List[Dict[str, Any]]) -> tuple:
        system_prompt = self._load_prompt_template()

        segment_lines = []
        for i, seg in enumerate(topic_segments):
            summary_text = seg.get("summary", "")
            segment_lines.append(f"Segment {i}: {summary_text}")

        segments_block = "\n".join(segment_lines)
        user_content = f"---BEGIN TOPIC SEGMENTS---\n{segments_block}\n---END TOPIC SEGMENTS---"
        return system_prompt, user_content

    def _print_distribution_report(
        self,
        similarities: List[float],
        topics: List[List[Dict[str, Any]]],
        segment_summaries: List[Dict[str, Any]],
    ) -> None:
        n = len(similarities)
        if n == 0:
            print("Only one segment — no similarity pairs to report.")
            return

        min_sim = min(similarities)
        max_sim = max(similarities)
        mean_sim = sum(similarities) / n

        boundaries = []
        for i, sim in enumerate(similarities):
            if sim < self._threshold:
                sid = segment_summaries[i].get("segment_id", f"segment_{i:03d}")
                boundaries.append(f"after {sid} ({sim:.2f})")

        boundary_str = ", ".join(boundaries) if boundaries else "none"
        print(f"\nSimilarity distribution across {n} pairs:")
        print(f"  min={min_sim:.2f}  max={max_sim:.2f}  mean={mean_sim:.2f}")
        print(f"  Boundaries (< {self._threshold}): {boundary_str}")
        print(f"  Topics detected: {len(topics)}")

    def _print_dry_run_plan(
        self,
        conversation_id: str,
        topics: List[List[Dict[str, Any]]],
    ) -> None:
        print(f"\n[DRY RUN] Topic plan for conversation: {conversation_id}")
        for i, topic in enumerate(topics):
            first_id = topic[0].get("segment_id", f"segment_{0:03d}")
            last_id = topic[-1].get("segment_id", f"segment_{len(topic) - 1:03d}")
            print(f"  Topic {i}: segments {first_id}–{last_id} ({len(topic)} segment(s))")
        print("  Re-run without --dry-run to summarize.")
