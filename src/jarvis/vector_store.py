"""Qdrant vector store client for JARVIS semantic retrieval.

Qdrant is used exclusively for vector retrieval. SQLite remains the source
of truth for structured summary records. Qdrant points carry only the
payload metadata needed for filtering; full records are fetched from SQLite.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


logger = logging.getLogger(__name__)

COLLECTION_NAME = "jarvis_summaries"


class VectorStore:
    """Manages the Qdrant collection for summary embeddings."""

    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize Qdrant client.

        Args:
            host: Qdrant server host.
            port: Qdrant server port.
        """
        self._client = QdrantClient(host=host, port=port)
        self._expected_dim: Optional[int] = None
        self._load_existing_dim()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(
        self,
        vector: List[float],
        summary_id: int,
        payload: Dict[str, Any],
    ) -> str:
        """Insert or update a vector point in Qdrant.

        Creates the collection lazily on the first call, using the vector
        dimension from that first embedding. Subsequent calls validate that
        the incoming vector matches the established dimension and fail clearly
        if it does not.

        Args:
            vector: Dense float vector to store.
            summary_id: SQLite row ID (stored in payload for cross-referencing).
            payload: Metadata dict stored alongside the vector.

        Returns:
            UUID string of the Qdrant point.

        Raises:
            ValueError: If the vector dimension does not match the collection.
            RuntimeError: If Qdrant is unreachable or returns an error.
        """
        dim = len(vector)
        self._ensure_collection(dim)
        self._validate_dimension(dim)

        point_id = str(uuid.uuid4())
        full_payload = {"summary_id": summary_id, **payload}

        try:
            self._client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=full_payload,
                    )
                ],
            )
        except Exception as e:
            raise RuntimeError(f"Qdrant upsert failed: {e}") from e

        logger.info(
            f"Upserted point {point_id} (summary_id={summary_id}, dim={dim})"
        )
        return point_id

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """Search for the top-k most similar summaries.

        Args:
            query_vector: Dense query vector.
            top_k: Number of results to return.

        Returns:
            List of (summary_id, score, qdrant_point_id) tuples ranked by
            descending similarity score.

        Raises:
            ValueError: If the query vector dimension does not match the collection.
            RuntimeError: If the collection does not exist or Qdrant errors.
        """
        dim = len(query_vector)

        if not self._collection_exists():
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Run at least one summarization with --persist first."
            )

        self._validate_dimension(dim)

        try:
            results = self._client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            raise RuntimeError(f"Qdrant search failed: {e}") from e

        hits = []
        for hit in results.points:
            summary_id = hit.payload.get("summary_id")
            if summary_id is not None:
                hits.append((int(summary_id), float(hit.score), str(hit.id)))

        logger.info(f"Search returned {len(hits)} results (top_k={top_k})")
        return hits

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def delete_points(self, point_ids: List[str]) -> None:
        """Delete Qdrant points by their UUIDs.

        Args:
            point_ids: List of Qdrant point UUID strings to delete.
        """
        if not point_ids or not self._collection_exists():
            return
        try:
            self._client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=qmodels.PointIdsList(points=point_ids),
            )
            logger.info(f"Deleted {len(point_ids)} Qdrant points")
        except Exception as e:
            logger.warning(f"Qdrant delete failed: {e}")

    def get_by_conversation(
        self, conversation_id: str
    ) -> List[Tuple[int, List[float], Dict[str, Any]]]:
        """Fetch all vectors for a conversation, ordered by chunk_index from payload.

        Uses Qdrant scroll+filter API. Returns [] if the collection doesn't exist
        or no points match.

        Args:
            conversation_id: The parent conversation ID to filter by.

        Returns:
            List of (summary_id, vector, payload) tuples ordered by chunk_index ASC.
        """
        if not self._collection_exists():
            return []

        results: List[Tuple[int, List[float], Dict[str, Any]]] = []
        offset = None

        try:
            while True:
                response, next_offset = self._client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="parent_conversation_id",
                                match=qmodels.MatchValue(value=conversation_id),
                            )
                        ]
                    ),
                    limit=100,
                    offset=offset,
                    with_vectors=True,
                    with_payload=True,
                )
                for point in response:
                    summary_id = point.payload.get("summary_id")
                    if summary_id is not None and point.vector is not None:
                        results.append(
                            (int(summary_id), list(point.vector), dict(point.payload))
                        )
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning(f"Qdrant scroll failed for conversation {conversation_id}: {e}")
            return []

        results.sort(key=lambda t: t[2].get("chunk_index", 0))
        logger.info(
            f"Fetched {len(results)} vectors from Qdrant for conversation {conversation_id}"
        )
        return results

    def _collection_exists(self) -> bool:
        """Check whether the collection exists in Qdrant."""
        collections = self._client.get_collections().collections
        return any(c.name == COLLECTION_NAME for c in collections)

    def _load_existing_dim(self) -> None:
        """Read the vector dimension from an existing collection, if any."""
        if not self._collection_exists():
            return
        info = self._client.get_collection(COLLECTION_NAME)
        self._expected_dim = info.config.params.vectors.size
        logger.debug(
            f"Loaded existing collection '{COLLECTION_NAME}' "
            f"with dim={self._expected_dim}"
        )

    def _ensure_collection(self, dim: int) -> None:
        """Create the collection if it does not yet exist.

        Args:
            dim: Vector dimension to use when creating the collection.
        """
        if self._collection_exists():
            return

        logger.info(
            f"Creating Qdrant collection '{COLLECTION_NAME}' with dim={dim}, "
            "distance=Cosine"
        )
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        self._expected_dim = dim

    def _validate_dimension(self, dim: int) -> None:
        """Raise clearly if dim does not match the established collection size.

        Args:
            dim: Dimension of the incoming vector.

        Raises:
            ValueError: If dim != expected_dim.
        """
        if self._expected_dim is not None and dim != self._expected_dim:
            raise ValueError(
                f"Vector dimension mismatch: collection '{COLLECTION_NAME}' expects "
                f"{self._expected_dim} dimensions but received {dim}. "
                f"Are you using the same embedding model that created the collection?"
            )
