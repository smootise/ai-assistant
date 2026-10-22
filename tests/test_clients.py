"""Factories consume the resolved profile and keep inference/embedding independent."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.clients import (
    build_memory_layer,
    create_embedding_client,
    create_ollama_client,
    create_vector_store,
)
from jarvis.config import load_config
from jarvis.profiles import ConfigurationError


@pytest.mark.parametrize("profile", ["personal", "mom"])
def test_resolved_profile_routes_both_stores(profile_repo: Path, profile: str) -> None:
    config = load_config(profile)
    with patch("jarvis.clients.VectorStore") as vectors, patch("jarvis.clients.EmbeddingClient"):
        memory = build_memory_layer(config)
    assert memory.store.db_path == config["profile_context"].db_path
    vectors.assert_called_once_with(
        host="localhost",
        port=6333,
        collection_name=config["qdrant_collection"],
        api_key=None,
    )


def test_identical_sqlite_ids_are_independent(profile_repo: Path) -> None:
    stores = []
    with patch("jarvis.clients.VectorStore"), patch("jarvis.clients.EmbeddingClient"):
        for profile in ("personal", "mom"):
            store = build_memory_layer(load_config(profile)).store
            store.insert_conversation({"conversation_id": "same-id", "title": profile})
            stores.append(store)
    assert stores[0].get_conversation("same-id")["title"] == "personal"
    assert stores[1].get_conversation("same-id")["title"] == "mom"


@pytest.mark.parametrize("embedding_url", [None, "http://synthetic-truenas:11434"])
def test_ollama_endpoints_default_or_split(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, embedding_url: str | None
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://synthetic-desktop:11434")
    if embedding_url:
        monkeypatch.setenv("EMBEDDING_OLLAMA_BASE_URL", embedding_url)
    config = load_config()
    with (
        patch("jarvis.clients.EmbeddingClient") as embeddings,
        patch("jarvis.clients.OllamaClient") as inference,
    ):
        create_embedding_client(config)
        create_ollama_client(config)
    embeddings.assert_called_once_with(
        model="qwen3-embedding",
        base_url=embedding_url or "http://synthetic-desktop:11434",
    )
    inference.assert_called_once_with(
        model="gemma4:31b",
        base_url="http://synthetic-desktop:11434",
        timeout=600,
    )


def test_legacy_dictionary_and_direct_vector_defaults() -> None:
    config = {"embedding_model": "qwen3-embedding", "ollama_base_url": "http://legacy:11434"}
    with patch("jarvis.clients.EmbeddingClient") as embeddings:
        create_embedding_client(config)
    embeddings.assert_called_once_with(model="qwen3-embedding", base_url="http://legacy:11434")
    server = MagicMock()
    server.collection_exists.return_value = False
    with patch("jarvis.vector_store.QdrantClient", return_value=server) as constructor:
        store = create_vector_store({"qdrant_host": "localhost", "qdrant_port": 6333})
    assert store.collection_name == "jarvis_fragments"
    constructor.assert_called_once_with(host="localhost", port=6333, api_key=None)


def test_remote_qdrant_url_and_key_reach_client(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://synthetic-qdrant.example:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "synthetic-key")
    server = MagicMock()
    server.collection_exists.return_value = False
    with patch("jarvis.vector_store.QdrantClient", return_value=server) as constructor:
        store = create_vector_store(load_config("mom"))
    constructor.assert_called_once_with(
        url="https://synthetic-qdrant.example:6333",
        api_key="synthetic-key",
    )
    assert store.collection_name == "jarvis_mom_fragments"


def test_factories_do_not_reload_configuration(profile_repo: Path) -> None:
    config = load_config("mom")
    with (
        patch("jarvis.config.load_config", side_effect=AssertionError("Reloaded config")),
        patch("jarvis.clients.VectorStore"),
        patch("jarvis.clients.EmbeddingClient"),
    ):
        assert build_memory_layer(config).store.db_path == config["profile_context"].db_path


def test_incomplete_non_personal_configuration_never_falls_back_to_personal_collection() -> None:
    with patch("jarvis.clients.VectorStore") as vectors:
        with pytest.raises(ConfigurationError, match="qdrant_collection"):
            create_vector_store({"profile_name": "mom"})
    vectors.assert_not_called()
