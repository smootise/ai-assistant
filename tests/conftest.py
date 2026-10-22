"""Isolated configuration and mandatory guards against production data/service access."""

import os
import socket
import sqlite3
from pathlib import Path
from typing import Iterator
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import url2pathname

import pytest

from jarvis import config
from jarvis import vector_store


@pytest.fixture(autouse=True)
def isolated_services_and_databases(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path_factory.getbasetemp().resolve()
    runtime = base / "isolated-config"
    runtime.mkdir(exist_ok=True)
    (runtime / "config.yaml").write_text("defaults: {}\n", encoding="utf-8")
    (runtime / "profiles.yaml").write_text("profiles: {personal: {}}\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", runtime)
    for name in os.environ:
        if name.startswith(
            ("JARVIS_", "OLLAMA_", "EMBEDDING_", "QDRANT_", "OPENAI_", "LOCAL_MODEL_")
        ):
            monkeypatch.delenv(name)
    connect = sqlite3.connect

    def temporary_connect(
        database: str | Path, *args: object, **kwargs: object
    ) -> sqlite3.Connection:
        if str(database) != ":memory:":
            value = str(database)
            if value.startswith("file:"):
                value = url2pathname(urlsplit(value).path)
            path = Path(value).resolve()
            if not path.is_relative_to(base):
                raise AssertionError("Tests may only open temporary SQLite databases")
        return connect(database, *args, **kwargs)

    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must mock service clients; live service access is forbidden")

    monkeypatch.setattr(sqlite3, "connect", temporary_connect)
    monkeypatch.setattr(socket.socket, "connect", unavailable)
    monkeypatch.setattr(socket.socket, "connect_ex", unavailable)
    monkeypatch.setattr(vector_store, "QdrantClient", unavailable)


@pytest.fixture
def profile_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    (tmp_path / "config.yaml").write_text("defaults: {}\n", encoding="utf-8")
    (tmp_path / "profiles.yaml").write_text(
        "profiles:\n"
        "  personal: {}\n"
        "  mom:\n"
        "    db_path: data/profiles/mom/jarvis.db\n"
        "    qdrant_collection: jarvis_mom_fragments\n"
        "    inbox_root: data/profiles/mom/inbox\n"
        "    output_root: data/profiles/mom/OUTPUTS\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        yield tmp_path
