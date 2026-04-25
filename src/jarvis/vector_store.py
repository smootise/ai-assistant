"""Qdrant vector store client for JARVIS semantic retrieval.

Qdrant indexes fragment vectors only. SQLite is the source of truth for all
structured records. Qdrant points carry a minimal payload for filtering;
full records are always reconstructed from SQLite by fragment_id.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


logger = logging.getLogger(__name__)

COLLECTION_NAME = "jarvis_fragments"


class VectorStore:
    """Manages the Qdrant collection for fragment embeddings."""

    def __init__(self, host: str = "localhost", port: int = 6333):
        self._client = QdrantClient(host=host, port=port)
        self._expected_dim: Optional[int] = None
        self._load_existing_dim()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(
        self,
        vector: List[float],
        fragment_id: str,
        payload: Dict[str, Any],
    ) -> str:
        """Insert or update a fragment vector point in Qdrant.

        Creates the collection lazily on the first call. Validates vector
        dimension on subsequent calls.

        Args:
            vector: Dense float vector to store.
            fragment_id: SQLite fragment_id (stored in payload for cross-referencing).
            payload: Minimal metadata dict stored alongside the vector.

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
        full_payload = {"fragment_id": fragment_id, **payload}

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

        logger.info(f"Upserted point {point_id} (fragment_id={fragment_id}, dim={dim})")
        return point_id

    def search(
        self, query_vector: List[float], top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """Search for the top-k most similar fragments.

        Args:
            query_vector: Dense query vector.
            top_k: Number of results to return.

        Returns:
            List of (fragment_id, score, qdrant_point_id) tuples ranked by
            descending similarity score.

        Raises:
            ValueError: If the query vector dimension does not match the collection.
            RuntimeError: If the collection does not exist or Qdrant errors.
        """
        dim = len(query_vector)

        if not self._collection_exists():
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
                "Run fragment-extracts with --persist --embed first."
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
            fragment_id = hit.payload.get("fragment_id")
            if fragment_id is not None:
                hits.append((str(fragment_id), float(hit.score), str(hit.id)))

        logger.info(f"Search returned {len(hits)} results (top_k={top_k})")
        return hits

    def delete_points(self, point_ids: List[str]) -> None:
        """Delete Qdrant points by their UUIDs."""
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collection_exists(self) -> bool:
        collections = self._client.get_collections().collections
        return any(c.name == COLLECTION_NAME for c in collections)

    def _load_existing_dim(self) -> None:
        if not self._collection_exists():
            return
        info = self._client.get_collection(COLLECTION_NAME)
        self._expected_dim = info.config.params.vectors.size
        logger.debug(
            f"Loaded existing collection '{COLLECTION_NAME}' "
            f"with dim={self._expected_dim}"
        )

    def _ensure_collection(self, dim: int) -> None:
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
        if self._expected_dim is not None and dim != self._expected_dim:
            raise ValueError(
                f"Vector dimension mismatch: collection '{COLLECTION_NAME}' expects "
                f"{self._expected_dim} dimensions but received {dim}. "
                f"Are you using the same embedding model that created the collection?"
            )
