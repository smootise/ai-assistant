"""Construct clients from an already resolved configuration; never reload global config."""

from typing import TYPE_CHECKING, Any, Mapping

from jarvis.embedder import EmbeddingClient
from jarvis.memory import MemoryLayer
from jarvis.ollama import OllamaClient
from jarvis.profiles import ConfigurationError
from jarvis.store import SummaryStore
from jarvis.vector_store import VectorStore

if TYPE_CHECKING:
    from jarvis.email_store import EmailStore
    from jarvis.email_processing_store import EmailProcessingStore


def create_email_processing_store(
    config: Mapping[str, Any], *, read_only: bool = False
) -> "EmailProcessingStore":
    from pathlib import Path
    from jarvis.email_processing_store import EmailProcessingStore

    return EmailProcessingStore(
        Path(config["db_path"]), config["profile_name"], read_only=read_only
    )


def create_email_store(config: Mapping[str, Any], *, read_only: bool = False) -> "EmailStore":
    from pathlib import Path
    from jarvis.email_store import EmailStore

    return EmailStore(Path(config["db_path"]), config["profile_name"], read_only=read_only)


def create_summary_store(config: Mapping[str, Any], *, read_only: bool = False) -> SummaryStore:
    if read_only:
        return SummaryStore(db_path=config["db_path"], read_only=True)
    return SummaryStore(db_path=config["db_path"])


def create_vector_store(config: Mapping[str, Any]) -> VectorStore:
    if "qdrant_collection" not in config and config.get("profile_name", "personal") != "personal":
        raise ConfigurationError("Resolved non-personal configuration requires qdrant_collection")
    options = {
        "collection_name": config.get("qdrant_collection", "jarvis_fragments"),
        "api_key": config.get("qdrant_api_key") or None,
    }
    if config.get("qdrant_url"):
        return VectorStore(url=config["qdrant_url"], **options)
    return VectorStore(
        host=config.get("qdrant_host", "localhost"),
        port=config.get("qdrant_port", 6333),
        **options,
    )


def create_embedding_client(config: Mapping[str, Any]) -> EmbeddingClient:
    return EmbeddingClient(
        model=config["embedding_model"],
        base_url=config.get("embedding_ollama_base_url") or config["ollama_base_url"],
    )


def create_ollama_client(config: Mapping[str, Any]) -> OllamaClient:
    return OllamaClient(
        model=config["local_model_name"],
        base_url=config["ollama_base_url"],
        timeout=config["ollama_timeout"],
    )


def build_memory_layer(config: Mapping[str, Any], *, read_only: bool = False) -> MemoryLayer:
    from jarvis.recovery import check_search_ready

    resolved = dict(config)
    return MemoryLayer(
        store=(
            create_summary_store(resolved, read_only=True)
            if read_only
            else create_summary_store(resolved)
        ),
        vector_store_factory=lambda: create_vector_store(resolved),
        embedder_factory=lambda: create_embedding_client(resolved),
        search_guard=lambda: check_search_ready(resolved),
    )
