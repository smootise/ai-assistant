"""Managed boundaries use only synthetic profiles and temporary filesystem objects."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.config import load_config
from jarvis.paths import PathBoundaryError, ProfilePaths, identifier, validate_tree
from jarvis.ingest.pipeline import ingest_chatgpt


@pytest.mark.parametrize("profile", ["personal", "mom"])
def test_profile_defaults_and_valid_overrides_are_read_only(
    profile_repo: Path, profile: str
) -> None:
    config = load_config(profile)
    paths = ProfilePaths.from_config(config)
    before = sorted(profile_repo.rglob("*"))
    assert paths.inbox() == config["profile_context"].chatgpt_inbox_dir
    assert paths.inbox(paths.context.inbox_root / "custom") == paths.context.inbox_root / "custom"
    assert paths.output("conv", "extracts") == paths.context.output_root / "conv/extracts"
    assert sorted(profile_repo.rglob("*")) == before


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "../other",
        "nested/name",
        "nested\\name",
        "/absolute",
        "C:relative",
        "C:\\absolute",
        "bad:stream",
        "CON",
        "nul.json",
        "COM1",
        "bad.",
        "bad ",
        "bad\x00id",
    ],
)
def test_unsafe_identifiers(value: str) -> None:
    with pytest.raises(PathBoundaryError):
        identifier(value)


@pytest.mark.parametrize("profile,other", [("personal", "mom"), ("mom", "personal")])
def test_cross_profile_paths_rejected_before_reading(
    profile_repo: Path, profile: str, other: str
) -> None:
    config = load_config(profile)
    paths = ProfilePaths.from_config(config)
    foreign = config["profiles"][other]
    with patch.object(Path, "open", side_effect=AssertionError("No file reads allowed")):
        for root in (foreign.inbox_root, foreign.output_root):
            with pytest.raises(PathBoundaryError):
                paths.inbox(root)
            with pytest.raises(PathBoundaryError, match="another profile"):
                paths.source(root / "export.json")
        with pytest.raises(PathBoundaryError, match="Database"):
            paths.source(foreign.db_path)


@pytest.mark.parametrize("value", ["../escape", "inside/../escape", "inside\\..\\escape"])
def test_managed_override_traversal_rejected(profile_repo: Path, value: str) -> None:
    paths = ProfilePaths.from_config(load_config())
    with pytest.raises(PathBoundaryError):
        paths.inbox(paths.context.inbox_root / value)


def symlink(link: Path, target: Path, directory: bool = False) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError:
        pytest.skip("Symlinks unavailable for this account")


def test_symlink_directory_escape(profile_repo: Path) -> None:
    paths = ProfilePaths.from_config(load_config())
    foreign = load_config("mom")["profile_context"].output_root
    foreign.mkdir(parents=True)
    link = paths.context.inbox_root / "escape"
    symlink(link, foreign, True)
    with pytest.raises(PathBoundaryError):
        paths.inbox(link)


def test_symlink_artifact_escape(profile_repo: Path) -> None:
    paths = ProfilePaths.from_config(load_config())
    outside = profile_repo / "external.json"
    outside.write_text('{"conversation_id": "conv"}', encoding="utf-8")
    directory = paths.output("conv")
    symlink(directory / "normalized.json", outside)
    with pytest.raises(PathBoundaryError):
        validate_tree(paths.context.output_root, directory, "conv")
    with pytest.raises(PathBoundaryError):
        paths.source(directory / "normalized.json")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_windows_junction_escape(profile_repo: Path) -> None:
    paths = ProfilePaths.from_config(load_config())
    foreign = load_config("mom")["profile_context"].output_root
    foreign.mkdir(parents=True)
    paths.context.inbox_root.mkdir(parents=True)
    link = paths.context.inbox_root / "junction"
    junction(link, foreign)
    with pytest.raises(PathBoundaryError):
        paths.inbox(link / "nested")


def junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode:
        pytest.skip("Junctions unavailable on this filesystem")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_junction_cannot_redirect_generated_files_over_original_source(profile_repo: Path) -> None:
    config = load_config()
    paths = ProfilePaths.from_config(config)
    source = paths.context.inbox_root / "originals/segment_000.json"
    source.parent.mkdir(parents=True)
    raw = json.loads(
        (Path(__file__).parent / "fixtures/chatgpt_tiny.json").read_text(encoding="utf-8")
    )
    raw["conversation_id"] = "collision"
    source.write_text(json.dumps(raw), encoding="utf-8")
    destination = paths.conversation("collision")
    junction(destination / "segments", source.parent)
    before = source.read_bytes(), source.stat().st_mtime_ns
    with pytest.raises(PathBoundaryError, match="overlaps"):
        ingest_chatgpt(source, paths.inbox(), False, config)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert not (destination / "normalized.json").exists()


def test_hard_linked_sources_rejected(profile_repo: Path) -> None:
    paths = ProfilePaths.from_config(load_config())
    source = profile_repo / "synthetic.json"
    source.write_text("{}", encoding="utf-8")
    alias = profile_repo / "alias.json"
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("Hard links unavailable")
    with pytest.raises(PathBoundaryError):
        paths.source(alias)


@pytest.mark.parametrize(
    "record",
    [
        {"segment_id": "../escape"},
        {"conversation_id": "different"},
        {"segment_index": -1},
        {"segment_index": True},
        {"fragment_index": "../escape"},
    ],
)
def test_artifact_identifiers_checked_before_processing(profile_repo: Path, record: dict) -> None:
    paths = ProfilePaths.from_config(load_config())
    directory = paths.output("conv")
    directory.mkdir(parents=True)
    artifact = directory / "record.json"
    artifact.write_text(json.dumps(record), encoding="utf-8")
    before = artifact.read_bytes(), artifact.stat().st_mtime_ns
    with pytest.raises(PathBoundaryError):
        validate_tree(paths.context.output_root, directory, "conv")
    assert (artifact.read_bytes(), artifact.stat().st_mtime_ns) == before


def test_direct_ingestion_cannot_bypass_profile_boundaries(profile_repo: Path) -> None:
    config = load_config("mom")
    foreign = load_config("personal")["profile_context"].output_root / "private.json"
    with patch.object(Path, "open", side_effect=AssertionError("No source should be read")):
        with pytest.raises(PathBoundaryError, match="another profile"):
            ingest_chatgpt(foreign, Path(config["chatgpt_inbox_dir"]), False, config)
    assert not config["profile_context"].inbox_root.exists()


@pytest.mark.parametrize(
    "field", ["output_root", "inbox_root", "db_path", "chatgpt_inbox_dir", "qdrant_collection"]
)
def test_inconsistent_aliases_are_rejected_instead_of_redirected(
    profile_repo: Path, field: str
) -> None:
    config = load_config("mom")
    config[field] = load_config("personal")[field]
    with pytest.raises(PathBoundaryError, match="disagrees"):
        ProfilePaths.from_config(config)
