"""Memory layer orchestrator for JARVIS.

Coordinates persistence to SQLite and vector indexing in Qdrant after
a summarization run. This is the only module that touches both stores.

Responsibility split:
- SQLite (SummaryStore)  → source of truth for structured summary records.
- Qdrant (VectorStore)   → vector retrieval index; payload metadata only.
- OUTPUTS folder         → raw JSON/Markdown artifacts.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from jarvis.embedder import EmbeddingClient, build_embedding_text
from jarvis.store import SummaryStore
from jarvis.vector_store import VectorStore


logger = logging.getLogger(__name__)


class MemoryLayer:
    """Persists a summary to SQLite and indexes it in Qdrant."""

    def __init__(
        self,
        store: SummaryStore,
        vector_store: VectorStore,
        embedder: EmbeddingClient,
    ):
        """Initialize memory layer.

        Args:
            store: SQLite summary store.
            vector_store: Qdrant vector store.
            embedder: Ollama embedding client.
        """
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder

    def persist(
        self,
        output_data: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> int:
        """Persist a summary to SQLite and index it in Qdrant.

        Steps:
        1. Insert summary record into SQLite → get summary_id.
        2. Build canonical embedding text from semantic fields.
        3. Embed with Ollama.
        4. Upsert vector into Qdrant with payload metadata.
        5. Update SQLite row with qdrant_point_id and embedded_at.

        Args:
            output_data: The summarization output dict.
            output_dir: Path to the OUTPUTS run directory (used to derive
                        artifact file paths stored in SQLite).

        Returns:
            SQLite summary_id of the persisted record.

        Raises:
            RuntimeError: On any hard failure in embedding or Qdrant upsert.
        """
        source_file = output_data.get("source_file", "")

        # Derive artifact paths from the output dir
        output_json_path: Optional[str] = None
        output_md_path: Optional[str] = None
        if output_dir is not None:
            stem = Path(source_file).stem
            output_json_path = str(output_dir / f"{stem}.json")
            output_md_path = str(output_dir / f"{stem}.md")

        # Step 1 — insert into SQLite
        summary_id = self.store.insert_summary(
            output_data=output_data,
            output_json_path=output_json_path,
            output_md_path=output_md_path,
        )

        # Step 2 — build canonical text
        embedding_text = build_embedding_text(output_data)
        logger.debug(f"Embedding text ({len(embedding_text)} chars):\n{embedding_text[:200]}…")

        # Step 3 — embed
        try:
            vector = self.embedder.embed(embedding_text)
        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Embedding failed for summary_id={summary_id}: {e}")
            raise

        # Step 4 — upsert into Qdrant
        payload = {
            "source_file": output_data.get("source_file"),
            "source_kind": output_data.get("source_kind"),
            "lang": output_data.get("lang"),
            "status": output_data.get("status"),
            "created_at": output_data.get("created_at"),
            "model": output_data.get("model"),
            "segment_id": output_data.get("segment_id"),
            "parent_conversation_id": output_data.get("parent_conversation_id"),
        }
        try:
            point_id = self.vector_store.upsert(
                vector=vector,
                summary_id=summary_id,
                payload=payload,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"Qdrant upsert failed for summary_id={summary_id}: {e}")
            raise

        # Step 5 — write back Qdrant point ID and embedding metadata to SQLite
        self.store.update_embedding(
            summary_id=summary_id,
            qdrant_point_id=point_id,
            embedding_model=self.embedder.model,
        )

        logger.info(
            f"Persisted summary_id={summary_id} "
            f"(source={source_file}, qdrant_point={point_id})"
        )
        return summary_id
