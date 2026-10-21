"""Synthetic profile registry and filesystem isolation tests."""

import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from jarvis.profiles import (
    ConfigurationError,
    require_personal_runtime,
    resolve_path,
    resolve_profiles,
)


def registry() -> dict[str, Any]:
    return {
        "profiles": {
            "personal": {},
            "mom": {
                "db_path": "data/profiles/mom/jarvis.db",
                "qdrant_collection": "jarvis_mom_fragments",
                "inbox_root": "data/profiles/mom/inbox",
                "output_root": "data/profiles/mom/OUTPUTS",
            },
        }
    }


def personal_defaults() -> dict[str, str]:
    return {
        "db_path": "data/jarvis.db",
        "qdrant_collection": "jarvis_fragments",
        "inbox_root": "inbox",
        "output_root": "OUTPUTS",
        "user_name": "",
    }


def test_defaults_are_frozen_and_do_not_create_storage(tmp_path: Path) -> None:
    profiles = resolve_profiles(registry(), tmp_path, personal_defaults(), {})
    assert profiles["personal"].db_path == tmp_path / "data" / "jarvis.db"
    assert profiles["mom"].qdrant_collection == "jarvis_mom_fragments"
    assert profiles["mom"].inbox_raw_dir == (
        tmp_path / "data/profiles/mom/inbox/ai_chat/chatgpt/raw"
    )
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(FrozenInstanceError):
        profiles["mom"].user_name = "changed"


def test_additional_profile_needs_only_a_registry_entry(tmp_path: Path) -> None:
    data = registry()
    data["profiles"]["project"] = {
        "db_path": "data/project/jarvis.db",
        "qdrant_collection": "project_fragments",
        "inbox_root": "data/project/inbox",
        "output_root": "data/project/OUTPUTS",
        "user_name": "Project owner",
    }
    assert resolve_profiles(data, tmp_path, personal_defaults(), {})["project"].user_name == (
        "Project owner"
    )


@pytest.mark.parametrize("field", ["db_path", "inbox_root", "output_root"])
def test_identical_storage_is_rejected(tmp_path: Path, field: str) -> None:
    data = registry()
    data["profiles"]["mom"][field] = personal_defaults()[field]
    with pytest.raises(ConfigurationError, match="personal.*mom") as error:
        resolve_profiles(data, tmp_path, personal_defaults(), {})
    assert str(tmp_path) in str(error.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "path"),
    [
        ("inbox_root", "OUTPUTS/nested"),
        ("output_root", "."),
        ("db_path", "inbox/mom.db"),
        ("output_root", "data"),
        ("db_path", "data/jarvis.db-wal"),
    ],
)
def test_nested_and_sidecar_conflicts(tmp_path: Path, field: str, path: str) -> None:
    data = registry()
    data["profiles"]["mom"][field] = path
    with pytest.raises(ConfigurationError):
        resolve_profiles(data, tmp_path, personal_defaults(), {})


def test_duplicate_collection_is_rejected(tmp_path: Path) -> None:
    data = registry()
    data["profiles"]["mom"]["qdrant_collection"] = "jarvis_fragments"
    with pytest.raises(ConfigurationError, match="personal.*mom.*jarvis_fragments"):
        resolve_profiles(data, tmp_path, personal_defaults(), {})


def test_effective_personal_environment_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="personal.*mom"):
        resolve_profiles(
            registry(),
            tmp_path,
            personal_defaults(),
            {"JARVIS_OUTPUT_ROOT": "data/profiles/mom/OUTPUTS"},
        )


@pytest.mark.parametrize("field", ["db_path", "inbox_root", "output_root", "qdrant_collection"])
def test_new_profiles_require_all_storage_fields(tmp_path: Path, field: str) -> None:
    data = registry()
    del data["profiles"]["mom"][field]
    with pytest.raises(ConfigurationError, match=f"mom.*missing.*{field}"):
        resolve_profiles(data, tmp_path, personal_defaults(), {})


@pytest.mark.parametrize(
    "entry",
    [None, [], {"typo": "value"}, {"db_path": ""}, {"qdrant_collection": "../bad"}],
)
def test_invalid_profile_entries(tmp_path: Path, entry: Any) -> None:
    data = registry()
    data["profiles"]["personal"] = entry
    with pytest.raises(ConfigurationError):
        resolve_profiles(data, tmp_path, personal_defaults(), {})


@pytest.mark.parametrize("name", ["../mom", "Mom", "", "mom/profile", 7])
def test_invalid_profile_names(tmp_path: Path, name: Any) -> None:
    data = registry()
    data["profiles"][name] = {}
    with pytest.raises(ConfigurationError, match="Profile names"):
        resolve_profiles(data, tmp_path, personal_defaults(), {})


def test_existing_database_hardlink_is_rejected_without_modification(tmp_path: Path) -> None:
    original = tmp_path / "personal.db"
    alias = tmp_path / "alias.db"
    original.write_bytes(b"synthetic sentinel, not a real user database")
    os.link(original, alias)
    data = registry()
    data["profiles"]["personal"]["db_path"] = str(original)
    data["profiles"]["mom"]["db_path"] = str(alias)
    with pytest.raises(ConfigurationError, match="personal.*mom"):
        resolve_profiles(data, tmp_path, personal_defaults(), {})
    assert original.read_bytes() == b"synthetic sentinel, not a real user database"


def test_symlink_alias_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "inbox"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable for this account")
    data = registry()
    data["profiles"]["mom"]["inbox_root"] = str(alias)
    with pytest.raises(ConfigurationError, match="personal.*mom"):
        resolve_profiles(data, tmp_path, personal_defaults(), {})


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_windows_junction_alias_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "inbox"
    target.mkdir()
    alias = tmp_path / "alias"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    try:
        data = registry()
        data["profiles"]["mom"]["inbox_root"] = str(alias)
        with pytest.raises(ConfigurationError, match="personal.*mom"):
            resolve_profiles(data, tmp_path, personal_defaults(), {})
    finally:
        alias.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="Windows case-insensitive paths")
def test_windows_case_variant_is_rejected(tmp_path: Path) -> None:
    data = registry()
    data["profiles"]["mom"]["inbox_root"] = "INBOX"
    with pytest.raises(ConfigurationError, match="personal.*mom"):
        resolve_profiles(data, tmp_path, personal_defaults(), {})


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-relative paths")
@pytest.mark.parametrize("path", ["C:relative", "\\relative"])
def test_windows_ambiguous_paths_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ConfigurationError):
        resolve_path(path, tmp_path, "test path")


def test_database_directory_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "data/jarvis.db").mkdir(parents=True)
    with pytest.raises(ConfigurationError, match="not a file"):
        resolve_profiles(registry(), tmp_path, personal_defaults(), {})


@pytest.mark.skipif(os.name != "nt", reason="Windows path aliases and reserved names")
@pytest.mark.parametrize("path", ["inbox.", "inbox ", "inbox:stream", "NUL/file", "CON.json"])
def test_windows_invalid_components_are_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ConfigurationError, match="invalid Windows"):
        resolve_path(path, tmp_path, "test path")


def test_non_directory_parent_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "data").write_text("synthetic file", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        resolve_profiles(registry(), tmp_path, personal_defaults(), {})


def test_runtime_guard_preserves_legacy_config_and_blocks_new_profiles(tmp_path: Path) -> None:
    require_personal_runtime({"db_path": str(tmp_path / "synthetic.db")})
    with pytest.raises(ConfigurationError, match="config show"):
        require_personal_runtime({"profile_name": "mom"})
    with pytest.raises(ConfigurationError, match="inspection-only"):
        require_personal_runtime({"qdrant_collection": "custom_fragments"})
    with pytest.raises(ConfigurationError, match="inspection-only"):
        require_personal_runtime({"repo_root": tmp_path, "inbox_root": tmp_path / "custom"})
