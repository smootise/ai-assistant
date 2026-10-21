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


@pytest.mark.parametrize("owner", ["personal", "mom"])
@pytest.mark.parametrize("same_profile", [True, False])
@pytest.mark.parametrize("field", ["inbox_root", "output_root"])
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant"])
def test_reserved_database_files_conflict_with_managed_roots(
    tmp_path: Path, owner: str, same_profile: bool, field: str, suffix: str, relation: str
) -> None:
    data = registry()
    other = owner if same_profile else ("mom" if owner == "personal" else "personal")
    database = tmp_path / "reserved" / "jarvis.db"
    reserved = Path(f"{database}{suffix}")
    root = {
        "equal": reserved,
        "ancestor": reserved.parent,
        "descendant": reserved / "nested",
    }[relation]
    data["profiles"][owner]["db_path"] = str(database)
    data["profiles"][other][field] = str(root)

    with pytest.raises(ConfigurationError, match="Storage conflict") as error:
        resolve_profiles(data, tmp_path, personal_defaults(), {})
    assert f"profile '{owner}'" in str(error.value)
    assert f"profile '{other}'" in str(error.value)
    assert str(root.resolve()) in str(error.value)
    conflicting_file = database if relation == "ancestor" else reserved
    assert str(conflicting_file.resolve()) in str(error.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("owner", ["personal", "mom"])
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
def test_database_files_cannot_be_ancestors_of_other_database_files(
    tmp_path: Path, owner: str, suffix: str
) -> None:
    data = registry()
    other = "mom" if owner == "personal" else "personal"
    database = tmp_path / "reserved" / "jarvis.db"
    reserved = Path(f"{database}{suffix}")
    nested_database = reserved / "nested.db"
    data["profiles"][owner]["db_path"] = str(database)
    data["profiles"][other]["db_path"] = str(nested_database)

    with pytest.raises(ConfigurationError, match="Storage conflict") as error:
        resolve_profiles(data, tmp_path, personal_defaults(), {})
    for name in (owner, other):
        assert f"profile '{name}'" in str(error.value)
    assert str(reserved.resolve()) in str(error.value)
    assert str(nested_database.resolve()) in str(error.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_existing_sidecar_directory_is_rejected(tmp_path: Path, suffix: str) -> None:
    reserved = tmp_path / f"data/jarvis.db{suffix}"
    reserved.mkdir(parents=True)
    with pytest.raises(ConfigurationError, match="not a file"):
        resolve_profiles(registry(), tmp_path, personal_defaults(), {})
    assert list(reserved.iterdir()) == []


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_sidecar_cannot_alias_its_own_database(tmp_path: Path, suffix: str) -> None:
    database = tmp_path / "personal.db"
    database.write_bytes(b"synthetic database sentinel")
    os.link(database, Path(f"{database}{suffix}"))
    data = registry()
    data["profiles"]["personal"]["db_path"] = str(database)
    with pytest.raises(ConfigurationError, match="Storage conflict"):
        resolve_profiles(data, tmp_path, personal_defaults(), {})
    assert database.read_bytes() == b"synthetic database sentinel"


def test_database_parents_and_similar_sibling_names_remain_available(tmp_path: Path) -> None:
    data = registry()
    data["profiles"]["personal"]["output_root"] = "data/jarvis.db-wal-outputs"
    data["profiles"]["mom"]["db_path"] = "data/mom.db"
    resolved = resolve_profiles(data, tmp_path, personal_defaults(), {})
    assert resolved["personal"].db_path == tmp_path / "data/jarvis.db"
    assert resolved["mom"].db_path == tmp_path / "data/mom.db"
    assert resolved["personal"].output_root == tmp_path / "data/jarvis.db-wal-outputs"
    assert list(tmp_path.iterdir()) == []
