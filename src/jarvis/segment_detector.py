"""Topic segment detection and summarization for JARVIS.

Groups consecutive chunk summaries into topic segments by measuring cosine
similarity between adjacent chunk summary embeddings.  When similarity drops
below the threshold a new segment starts.  Each segment is then summarized
with a single LLM call.
"""

import json
import logging
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


def _build_embedding_text(chunk_summary: Dict[str, Any]) -> str:
    """Build the text to embed from a chunk summary dict."""
    parts = [chunk_summary.get("summary", "")]
    for bullet in chunk_summary.get("bullets", []):
        parts.append(bullet)
    return " ".join(p for p in parts if p)


class SegmentDetector:
    """Detects topic boundaries and summarizes each segment."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        ollama_client: OllamaClient,
        prompts_dir: str,
        schema: str,
        schema_version: str,
        threshold: float = 0.65,
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
        """Full pipeline: load embeddings → detect boundaries → summarize segments.

        Args:
            conversation_id: ID of the conversation to process.
            output_root: Root directory where OUTPUTS are stored.
            dry_run: If True, detect boundaries but skip LLM summarization.
            run_id: Optional run identifier to propagate to output documents.

        Returns:
            List of (output_dir, output_data) tuples, one per segment.
            Empty list when dry_run=True.

        Raises:
            FileNotFoundError: If no chunk summary files are found.
        """
        chunk_summaries_dir = output_root / conversation_id / "chunk_summaries"
        if not chunk_summaries_dir.exists():
            raise FileNotFoundError(
                f"Chunk summaries directory not found: {chunk_summaries_dir}. "
                "Run 'summarize-chunks' first."
            )

        chunk_files = sorted(chunk_summaries_dir.glob("*.json"))
        if not chunk_files:
            raise FileNotFoundError(
                f"No chunk summary JSON files found in {chunk_summaries_dir}."
            )

        chunk_summaries = []
        for f in chunk_files:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("source_kind") == "ai_chat_chunk":
                chunk_summaries.append(data)

        if not chunk_summaries:
            raise FileNotFoundError(
                f"No ai_chat_chunk summaries found in {chunk_summaries_dir}."
            )

        chunk_summaries.sort(key=lambda d: d.get("chunk_index", 0))

        # Warn about stale segment files
        segment_summaries_dir = output_root / conversation_id / "segment_summaries"
        if segment_summaries_dir.exists():
            existing = list(segment_summaries_dir.glob("segment_*.json"))
            if existing:
                logger.warning(
                    f"{len(existing)} existing segment files found — old files with higher "
                    "indices may become stale if segment count changes."
                )

        vectors = self._load_embeddings(conversation_id, chunk_summaries)
        segments, similarities = self.detect_segments(chunk_summaries, vectors)

        self._print_distribution_report(similarities, segments, chunk_summaries)

        if dry_run:
            self._print_dry_run_plan(conversation_id, segments)
            return []

        results = []
        for seg_idx, seg_chunks in enumerate(segments):
            output_dir, output_data = self.summarize_segment(
                segment_chunks=seg_chunks,
                segment_index=seg_idx,
                conversation_id=conversation_id,
                segment_summaries_dir=segment_summaries_dir,
                run_id=run_id,
            )
            results.append((output_dir, output_data))

        return results

    def detect_segments(
        self,
        chunk_summaries: List[Dict[str, Any]],
        vectors: List[List[float]],
    ) -> Tuple[List[List[Dict[str, Any]]], List[float]]:
        """Group chunks into topic segments based on cosine similarity.

        Args:
            chunk_summaries: Ordered list of chunk summary dicts.
            vectors: Corresponding embedding vectors (same order).

        Returns:
            Tuple of (segments, similarities) where segments is a list of chunk
            groups and similarities is the list of consecutive pair similarities.
        """
        if not chunk_summaries:
            return [], []

        if len(chunk_summaries) == 1:
            return [chunk_summaries], []

        similarities: List[float] = []
        segments: List[List[Dict[str, Any]]] = [[chunk_summaries[0]]]
        seg_num = 0

        for i in range(1, len(chunk_summaries)):
            sim = _cosine_similarity(vectors[i - 1], vectors[i])
            similarities.append(sim)

            if sim < self._threshold:
                chunk_id = chunk_summaries[i - 1].get("chunk_id", f"chunk_{i - 1:03d}")
                logger.info(
                    f"Boundary after {chunk_id} (similarity={sim:.3f} < "
                    f"threshold={self._threshold}) → segment {seg_num + 1}"
                )
                seg_num += 1
                segments.append([])

            segments[-1].append(chunk_summaries[i])

        return segments, similarities

    def summarize_segment(
        self,
        segment_chunks: List[Dict[str, Any]],
        segment_index: int,
        conversation_id: str,
        segment_summaries_dir: Path,
        run_id: Optional[str] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Summarize one segment with the LLM.

        Args:
            segment_chunks: Chunk summary dicts that form this segment.
            segment_index: Zero-based index of this segment.
            conversation_id: Parent conversation ID.
            segment_summaries_dir: Directory to write output files into.
            run_id: Optional run identifier.

        Returns:
            Tuple of (output_dir, output_data).
        """
        prompt = self._build_segment_prompt(segment_chunks)
        raw = self._ollama.generate(prompt)
        parsed = self._ollama.parse_json_response(raw)

        first_chunk_id = segment_chunks[0].get("chunk_id", "c000")
        last_chunk_id = segment_chunks[-1].get("chunk_id", f"c{len(segment_chunks) - 1:03d}")
        chunk_range = f"{first_chunk_id}-{last_chunk_id}"

        output_data = {
            **parsed,
            "source_file": f"segment_{segment_index:03d}.json",
            "source_kind": "ai_chat_segment",
            "segment_index": segment_index,
            "segment_chunk_range": chunk_range,
            "parent_conversation_id": conversation_id,
            "provider": "local",
            "model": self._ollama.model,
            "schema": self._schema,
            "schema_version": self._schema_version,
        }
        if run_id:
            output_data["run_id"] = run_id

        writer = OutputWriter(
            output_root=str(segment_summaries_dir),
            use_timestamp=False,
        )
        output_dir = writer.write_outputs(
            summary_data=output_data,
            source_file=f"segment_{segment_index:03d}.json",
        )
        logger.info(
            f"Segment {segment_index} summarized → {output_dir} "
            f"(chunks {chunk_range})"
        )
        return output_dir, output_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_embeddings(
        self,
        conversation_id: str,
        chunk_summaries: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """Return vectors for all chunk summaries.

        Fetches from Qdrant if available and count matches; otherwise embeds
        on the fly.

        Args:
            conversation_id: Used to query Qdrant.
            chunk_summaries: Chunk summary dicts (used for on-the-fly embedding).

        Returns:
            List of float vectors in the same order as chunk_summaries.
        """
        if self._vector_store is not None:
            qdrant_results = self._vector_store.get_by_conversation(conversation_id)
            if len(qdrant_results) == len(chunk_summaries):
                logger.info(
                    f"Reusing {len(qdrant_results)} vectors from Qdrant"
                )
                return [vec for _, vec, _ in qdrant_results]
            elif qdrant_results:
                logger.warning(
                    f"Qdrant returned {len(qdrant_results)} vectors but expected "
                    f"{len(chunk_summaries)} — falling back to on-the-fly embedding"
                )

        logger.info(f"Embedding {len(chunk_summaries)} chunk summaries on the fly")
        return [
            self._embedder.embed(_build_embedding_text(cs))
            for cs in chunk_summaries
        ]

    def _load_prompt_template(self) -> str:
        """Load the segment summarization prompt template (cached)."""
        if self._prompt_template is None:
            prompt_path = self._prompts_dir / "summarize_segment.md"
            self._prompt_template = prompt_path.read_text(encoding="utf-8")
        return self._prompt_template

    def _build_segment_prompt(self, segment_chunks: List[Dict[str, Any]]) -> str:
        """Build the full prompt for segment summarization.

        Args:
            segment_chunks: Chunk summary dicts for this segment.

        Returns:
            Complete prompt string.
        """
        template = self._load_prompt_template()

        chunk_lines = []
        for i, chunk in enumerate(segment_chunks):
            summary_text = chunk.get("summary", "")
            chunk_lines.append(f"Chunk {i}: {summary_text}")

        chunks_block = "\n".join(chunk_lines)
        injection = (
            f"\n\n---BEGIN SEGMENT CHUNKS---\n{chunks_block}\n---END SEGMENT CHUNKS---"
        )
        return template + injection

    def _print_distribution_report(
        self,
        similarities: List[float],
        segments: List[List[Dict[str, Any]]],
        chunk_summaries: List[Dict[str, Any]],
    ) -> None:
        """Print similarity distribution statistics to stdout."""
        n = len(similarities)
        if n == 0:
            print("Only one chunk — no similarity pairs to report.")
            return

        min_sim = min(similarities)
        max_sim = max(similarities)
        mean_sim = sum(similarities) / n

        boundaries = []
        for i, sim in enumerate(similarities):
            if sim < self._threshold:
                cid = chunk_summaries[i].get("chunk_id", f"chunk_{i:03d}")
                boundaries.append(f"after {cid} ({sim:.2f})")

        boundary_str = ", ".join(boundaries) if boundaries else "none"
        print(f"\nSimilarity distribution across {n} pairs:")
        print(f"  min={min_sim:.2f}  max={max_sim:.2f}  mean={mean_sim:.2f}")
        print(f"  Boundaries (< {self._threshold}): {boundary_str}")
        print(f"  Segments detected: {len(segments)}")

    def _print_dry_run_plan(
        self,
        conversation_id: str,
        segments: List[List[Dict[str, Any]]],
    ) -> None:
        """Print the dry-run segment plan to stdout."""
        print(f"\n[DRY RUN] Segment plan for conversation: {conversation_id}")
        for i, seg in enumerate(segments):
            first_id = seg[0].get("chunk_id", f"chunk_{0:03d}")
            last_id = seg[-1].get("chunk_id", f"chunk_{len(seg) - 1:03d}")
            print(f"  Segment {i}: chunks {first_id}–{last_id} ({len(seg)} chunk(s))")
        print("  Re-run without --dry-run to summarize.")
