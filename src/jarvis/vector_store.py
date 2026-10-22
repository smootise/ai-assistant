"""Qdrant vector store client for JARVIS semantic retrieval.

Qdrant indexes fragment vectors only. SQLite is the source of truth for all
structured records. Qdrant points carry a minimal payload for filtering;
full records are always reconstructed from SQLite by fragment_id.
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from jarvis.endpoints import validate_qdrant_endpoint

logger = logging.getLogger(__name__)

COLLECTION_NAME = "jarvis_fragments"


class VectorStore:
    """Manages the Qdrant collection for fragment embeddings."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str = COLLECTION_NAME,
        *,
        url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        validate_qdrant_endpoint(url, host, port)
        if not isinstance(collection_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", collection_name
        ):
            raise ValueError("Invalid Qdrant collection name")
        self.collection_name = collection_name
        self._expected_dim: Optional[int] = None
        try:
            if url:
                self._client = QdrantClient(url=url, api_key=api_key)
            else:
                self._client = QdrantClient(
                    host=host if host is not None else "localhost",
                    port=port if port is not None else 6333,
                    api_key=api_key,
                )
            self._load_existing_dim()
        except Exception:
            raise RuntimeError("Qdrant connection or collection lookup failed") from None

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
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=full_payload,
                    )
                ],
            )
        except Exception:
            raise RuntimeError("Qdrant upsert failed") from None

        logger.info(f"Upserted point {point_id} (fragment_id={fragment_id}, dim={dim})")
        return point_id

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, str]]:
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
                f"Qdrant collection '{self.collection_name}' does not exist. "
                "Run fragment-extracts with --persist --embed first."
            )

        if self._expected_dim is None:
            self._load_existing_dim()
        self._validate_dimension(dim)

        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception:
            raise RuntimeError("Qdrant search failed") from None

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
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=point_ids),
                wait=True,
            )
            logger.info(f"Deleted {len(point_ids)} Qdrant points")
        except Exception:
            raise RuntimeError("Qdrant delete failed") from None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collection_exists(self) -> bool:
        try:
            return self._client.collection_exists(collection_name=self.collection_name)
        except Exception:
            raise RuntimeError("Qdrant collection lookup failed") from None

    def _load_existing_dim(self) -> None:
        if not self._collection_exists():
            return
        try:
            info = self._client.get_collection(collection_name=self.collection_name)
            self._expected_dim = info.config.params.vectors.size
        except Exception:
            raise RuntimeError("Qdrant collection dimension lookup failed") from None
        logger.debug(
            f"Loaded existing collection '{self.collection_name}' " f"with dim={self._expected_dim}"
        )

    def _ensure_collection(self, dim: int) -> None:
        if self._collection_exists():
            if self._expected_dim is None:
                self._load_existing_dim()
            return

        logger.info(
            f"Creating Qdrant collection '{self.collection_name}' with dim={dim}, "
            "distance=Cosine"
        )
        try:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )
        except Exception:
            raise RuntimeError("Qdrant collection creation failed") from None
        self._expected_dim = dim

    def _validate_dimension(self, dim: int) -> None:
        if self._expected_dim is not None and dim != self._expected_dim:
            raise ValueError(
                f"Vector dimension mismatch: collection '{self.collection_name}' expects "
                f"{self._expected_dim} dimensions but received {dim}. "
                f"Are you using the same embedding model that created the collection?"
            )
