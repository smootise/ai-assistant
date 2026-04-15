"""Ollama embedding client and canonical text builder for JARVIS."""

import logging
from typing import Any, Dict, List

import requests


logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Calls the Ollama /api/embed endpoint to produce dense vectors."""

    def __init__(
        self,
        model: str = "qwen3-embedding",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize embedding client.

        Args:
            model: Ollama embedding model name.
            base_url: Ollama server base URL.
        """
        self.model = model
        self.embed_url = f"{base_url}/api/embed"

    def embed(self, text: str) -> List[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            Dense float vector.

        Raises:
            ConnectionError: If Ollama is unreachable.
            RuntimeError: If Ollama returns an error or an unexpected response.
        """
        payload = {"model": self.model, "input": text}

        logger.debug(f"Embedding text ({len(text)} chars) with model={self.model}")

        try:
            response = requests.post(self.embed_url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Ollama unreachable at {self.embed_url}. Is Ollama running?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError("Ollama embedding request timed out") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama embedding request failed: {e}") from e

        result = response.json()

        if "error" in result:
            raise RuntimeError(f"Ollama embedding error: {result['error']}")

        # Ollama /api/embed returns {"embeddings": [[...], ...]}
        embeddings = result.get("embeddings")
        if not embeddings or not isinstance(embeddings, list) or not embeddings[0]:
            raise RuntimeError(
                f"Unexpected embedding response shape: {list(result.keys())}"
            )

        vector = embeddings[0]
        logger.debug(f"Received vector of dimension {len(vector)}")
        return vector


def build_embedding_text(output_data: Dict[str, Any]) -> str:
    """Build a clean canonical text blob from summary semantic fields.

    The text is built from the human-readable fields only — no raw JSON.
    Original language is preserved; no translation is applied.

    Layout:
        [source_kind] [source_file]

        [summary]

        Key points:
        - bullet 1
        ...

        Action items:
        - action 1
        ...

    Args:
        output_data: Summarization output dict (matches OUTPUTS.md schema).

    Returns:
        A single UTF-8 string ready to be embedded.
    """
    parts: List[str] = []

    # Context header
    source_kind = output_data.get("source_kind", "")
    source_file = output_data.get("source_file", "")
    if source_kind or source_file:
        parts.append(f"{source_kind} {source_file}".strip())

    # Summary
    summary = (output_data.get("summary") or "").strip()
    if summary:
        parts.append(summary)

    # Bullets
    bullets = output_data.get("bullets") or []
    if bullets:
        bullet_lines = "\n".join(f"- {b}" for b in bullets)
        parts.append(f"Key points:\n{bullet_lines}")

    # Action items
    action_items = output_data.get("action_items") or []
    if action_items:
        action_lines = "\n".join(f"- {a}" for a in action_items)
        parts.append(f"Action items:\n{action_lines}")

    return "\n\n".join(parts)
