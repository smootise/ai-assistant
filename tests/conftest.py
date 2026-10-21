"""Opt-in isolated configuration fixtures; never read the repository's .env."""

import os
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from jarvis import config


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
