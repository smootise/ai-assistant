"""Construct clients from an already resolved configuration; never reload global config."""

from typing import Any, Mapping

from jarvis.embedder import EmbeddingClient
from jarvis.memory import MemoryLayer
from jarvis.ollama import OllamaClient
from jarvis.profiles import ConfigurationError
from jarvis.store import SummaryStore
from jarvis.vector_store import VectorStore


def create_summary_store(config: Mapping[str, Any]) -> SummaryStore:
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


def build_memory_layer(config: Mapping[str, Any]) -> MemoryLayer:
    return MemoryLayer(
        store=create_summary_store(config),
        vector_store=create_vector_store(config),
        embedder=create_embedding_client(config),
    )
