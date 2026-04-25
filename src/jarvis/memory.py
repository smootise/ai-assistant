"""Memory layer orchestrator for JARVIS.

Coordinates entity-level persistence to SQLite and fragment-level vector
indexing in Qdrant. This is the only module that touches both stores.

Responsibility split:
- SQLite (SummaryStore)  → relational source of truth for all records.
- Qdrant (VectorStore)   → fragment vector index only; minimal payload.
- OUTPUTS / inbox        → raw JSON/Markdown artifacts (stays on disk).
"""

import logging
from typing import Any, Dict, List, Optional

from jarvis.embedder import EmbeddingClient
from jarvis.store import SummaryStore
from jarvis.vector_store import VectorStore


logger = logging.getLogger(__name__)


class MemoryLayer:
    """Persists pipeline records to SQLite; indexes fragment vectors in Qdrant."""

    def __init__(
        self,
        store: SummaryStore,
        vector_store: VectorStore,
        embedder: EmbeddingClient,
    ):
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Ingest-stage persistence
    # ------------------------------------------------------------------

    def persist_source_file(self, data: Dict[str, Any]) -> str:
        """Insert source file metadata. Returns source_file_id."""
        fid = self.store.insert_source_file(data)
        logger.debug(f"Source file persisted: {fid} ({data['source_kind']})")
        return fid

    def persist_conversation(self, data: Dict[str, Any]) -> None:
        """Insert conversation metadata row."""
        self.store.insert_conversation(data)
        logger.debug(f"Conversation persisted: {data['conversation_id']}")

    def persist_segment(self, data: Dict[str, Any]) -> str:
        """Insert a segment row (with full segment_text). Returns segment_id."""
        sid = self.store.insert_segment(data)
        logger.debug(f"Segment persisted: {sid}")
        return sid

    # ------------------------------------------------------------------
    # Summarize-segments persistence
    # ------------------------------------------------------------------

    def persist_segment_summary(self, data: Dict[str, Any]) -> str:
        """Insert a segment summary row. Returns segment_summary_id."""
        ss_id = self.store.insert_segment_summary(data)
        logger.debug(f"Segment summary persisted: {ss_id}")
        return ss_id

    # ------------------------------------------------------------------
    # Topic persistence
    # ------------------------------------------------------------------

    def persist_topic_summary(
        self, data: Dict[str, Any], segment_ids: Optional[List[str]] = None
    ) -> str:
        """Insert topic summary + segment mapping rows. Returns topic_id."""
        topic_id = self.store.insert_topic_summary(data)
        if segment_ids:
            self.store.insert_topic_segments(topic_id, segment_ids)
        logger.debug(f"Topic summary persisted: {topic_id}")
        return topic_id

    # ------------------------------------------------------------------
    # Extract persistence
    # ------------------------------------------------------------------

    def persist_extract_with_statements(self, output_data: Dict[str, Any]) -> str:
        """Insert extract row + all statement rows in one logical operation.

        Idempotent: if the extract row already exists (segment_id UNIQUE),
        both insert_extract and insert_statements are no-ops.

        Returns the extract_id.
        """
        extract_id = self.store.insert_extract(output_data)
        statements = output_data.get("statements", [])
        if statements:
            self.store.insert_statements(extract_id, statements)
        logger.debug(
            f"Extract persisted: {extract_id} ({len(statements)} statements)"
        )
        return extract_id

    # ------------------------------------------------------------------
    # Fragment persistence
    # ------------------------------------------------------------------

    def persist_fragment_with_links(
        self, output_data: Dict[str, Any], extract_id: str
    ) -> str:
        """Insert fragment row + statement link rows.

        Idempotent: INSERT OR IGNORE on (extract_id, fragment_index).

        Args:
            output_data: Fragment output_data dict from Fragmenter.
            extract_id: Parent extract_id (required — fragmenter knows its extract).

        Returns the fragment_id.
        """
        frag_data: Dict[str, Any] = {
            "extract_id": extract_id,
            "fragment_index": output_data["fragment_index"],
            "title": output_data.get("title"),
            "retrieval_text": output_data.get("text", ""),
            "status": output_data.get("status", "ok"),
        }
        if output_data.get("created_at"):
            frag_data["created_at"] = output_data["created_at"]
        fragment_id = self.store.insert_fragment(frag_data)

        # Build statement_ids from linked statements (using deterministic IDs)
        statements = output_data.get("statements", [])
        statement_ids = [
            f"{extract_id}_st{s['statement_index']:04d}"
            for s in statements
            if isinstance(s.get("statement_index"), int)
        ]
        if statement_ids:
            self.store.insert_fragment_links(fragment_id, statement_ids)

        logger.debug(
            f"Fragment persisted: {fragment_id} "
            f"({len(statement_ids)} statement links)"
        )
        return fragment_id

    # ------------------------------------------------------------------
    # Fragment Qdrant indexing
    # ------------------------------------------------------------------

    def index_fragment_in_qdrant(
        self, fragment_id: str, output_data: Dict[str, Any]
    ) -> str:
        """Embed retrieval_text and upsert fragment into Qdrant.

        Writes qdrant_point_id back to the fragments table on success.

        Args:
            fragment_id: SQLite fragment_id of an already-persisted fragment.
            output_data: Fragment output_data (used for embedding text + payload).

        Returns:
            Qdrant point_id of the upserted vector.

        Raises:
            RuntimeError: On embedding or Qdrant upsert failure.
        """
        retrieval_text = (output_data.get("text") or "").strip()
        logger.debug(
            f"Embedding fragment {fragment_id} ({len(retrieval_text)} chars)"
        )

        try:
            vector = self.embedder.embed(retrieval_text)
        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Embedding failed for fragment {fragment_id}: {e}")
            raise

        payload = {
            "fragment_id": fragment_id,
            "parent_conversation_id": output_data.get("parent_conversation_id"),
            "segment_id": output_data.get("segment_id"),
            "conversation_date": output_data.get("conversation_date"),
        }

        try:
            point_id = self.vector_store.upsert(
                vector=vector,
                fragment_id=fragment_id,
                payload=payload,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"Qdrant upsert failed for fragment {fragment_id}: {e}")
            raise

        self.store.update_fragment_embedding(
            fragment_id=fragment_id,
            qdrant_point_id=point_id,
            embedding_model=self.embedder.model,
        )
        logger.info(
            f"Indexed fragment {fragment_id} in Qdrant (point={point_id})"
        )
        return point_id
