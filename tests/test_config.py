"""Read-only configuration precedence, inspection, and validation tests."""

import os
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from jarvis.config import format_config, load_config
from jarvis.profiles import ConfigurationError


def test_personal_defaults_and_no_environment_mutation(profile_repo: Path) -> None:
    before = dict(os.environ)
    paths_before = sorted(profile_repo.rglob("*"))
    config = load_config()
    assert config["profile_name"] == "personal"
    assert config["db_path"] == str(profile_repo / "data/jarvis.db")
    assert config["output_root"] == str(profile_repo / "OUTPUTS")
    assert config["qdrant_collection"] == "jarvis_fragments"
    assert dict(os.environ) == before
    assert sorted(profile_repo.rglob("*")) == paths_before


def test_profile_selection_precedence(profile_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (profile_repo / ".env").write_text("JARVIS_PROFILE=mom\n", encoding="utf-8")
    assert load_config()["profile_name"] == "mom"
    monkeypatch.setenv("JARVIS_PROFILE", "personal")
    assert load_config()["profile_name"] == "personal"
    assert load_config("mom")["profile_name"] == "mom"
    monkeypatch.setenv("JARVIS_PROFILE", "unknown")
    assert load_config("personal")["profile_name"] == "personal"


def test_personal_environment_overrides_do_not_leak_to_mom(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (profile_repo / ".env").write_text(
        "JARVIS_DB_PATH=custom/personal.db\n"
        "JARVIS_OUTPUT_ROOT=personal-artifacts\n"
        "JARVIS_USER_NAME=Personal owner\n"
        "OLLAMA_BASE_URL=http://shared-ollama:11434\n"
        "EMBEDDING_MODEL=shared-embedding\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("JARVIS_DB_PATH", "environment/personal.db")
    personal = load_config()
    mom = load_config("mom")
    assert personal["db_path"] == str(profile_repo / "environment/personal.db")
    assert personal["output_root"] == str(profile_repo / "personal-artifacts")
    assert personal["user_name"] == "Personal owner"
    assert mom["db_path"] == str(profile_repo / "data/profiles/mom/jarvis.db")
    assert mom["output_root"] == str(profile_repo / "data/profiles/mom/OUTPUTS")
    assert mom["user_name"] == ""
    assert mom["qdrant_collection"] == "jarvis_mom_fragments"
    assert personal["ollama_base_url"] == mom["ollama_base_url"]
    assert personal["embedding_model"] == mom["embedding_model"] == "shared-embedding"
    assert "JARVIS_OUTPUT_ROOT" not in os.environ


def test_personal_profile_values_override_config_defaults(profile_repo: Path) -> None:
    (profile_repo / "config.yaml").write_text(
        "defaults:\n  memory:\n    db_path: old/db.sqlite\n"
        "  outputs:\n    root: old/outputs\n  user_name: Old name\n",
        encoding="utf-8",
    )
    assert load_config()["db_path"] == str(profile_repo / "old/db.sqlite")
    registry_path = profile_repo / "profiles.yaml"
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    data["profiles"]["personal"] = {"db_path": "registered/db.sqlite", "user_name": "Owner"}
    registry_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = load_config()
    assert config["db_path"] == str(profile_repo / "registered/db.sqlite")
    assert config["output_root"] == str(profile_repo / "old/outputs")
    assert config["user_name"] == "Owner"


def test_resolution_is_independent_of_working_directory(
    profile_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    config = load_config("mom")
    assert config["repo_root"] == profile_repo
    assert config["prompts_dir"] == str(profile_repo / "prompts")
    assert config["samples_dir"] == str(profile_repo / "samples")
    assert config["db_path"] == str(profile_repo / "data/profiles/mom/jarvis.db")


def test_absolute_personal_paths_are_preserved(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    absolute = profile_repo / "external-personal.db"
    monkeypatch.setenv("JARVIS_DB_PATH", str(absolute))
    assert load_config()["db_path"] == str(absolute)


@pytest.mark.parametrize("selection", ["unknown", "", "Mom", "../mom"])
def test_invalid_selection_fails_without_fallback(profile_repo: Path, selection: str) -> None:
    with pytest.raises(ConfigurationError):
        load_config(selection)
    assert not (profile_repo / "data").exists()


@pytest.mark.parametrize("filename", ["profiles.yaml", "config.yaml"])
@pytest.mark.parametrize("contents", ["[]", "[invalid", "defaults: 1", "a: 1\na: 2"])
def test_malformed_or_duplicate_yaml_fails(
    profile_repo: Path, filename: str, contents: str
) -> None:
    (profile_repo / filename).write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config()
    assert not (profile_repo / "data").exists()


def test_missing_registry_fails_without_creating_a_replacement(profile_repo: Path) -> None:
    registry = profile_repo / "profiles.yaml"
    registry.unlink()
    with pytest.raises(ConfigurationError, match="Required configuration file"):
        load_config()
    assert not registry.exists()


def test_empty_registry_is_invalid(profile_repo: Path) -> None:
    (profile_repo / "profiles.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config()


def test_conflicting_inactive_profiles_are_rejected(profile_repo: Path) -> None:
    registry = profile_repo / "profiles.yaml"
    data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    data["profiles"]["project"] = dict(data["profiles"]["mom"])
    registry.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mom.*project"):
        load_config("personal")


def test_duplicate_nested_profile_fields_are_rejected(profile_repo: Path) -> None:
    (profile_repo / "profiles.yaml").write_text(
        "profiles:\n  personal:\n    db_path: first.db\n    db_path: second.db\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unique string keys"):
        load_config()


def test_registry_requires_personal(profile_repo: Path) -> None:
    (profile_repo / "profiles.yaml").write_text("profiles: {}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="register the default 'personal'"):
        load_config()


@pytest.mark.parametrize(
    ("variable", "value"),
    [("QDRANT_PORT", "invalid"), ("QDRANT_PORT", "70000"), ("OLLAMA_TIMEOUT", "0")],
)
def test_invalid_shared_numbers_fail_safely(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ConfigurationError, match=variable):
        load_config()


def test_display_uses_allowlist_and_redacts_url_credentials(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    monkeypatch.setenv(
        "OLLAMA_BASE_URL", "http://alice:password@localhost:11434/?token=hidden#secret"
    )
    monkeypatch.setenv("QDRANT_HOST", "bob:credential@localhost")
    config = load_config("mom")
    config["unexpected_secret"] = "do-not-display"
    output = format_config(config)
    for secret in (
        "secret-api-key",
        "password",
        "alice",
        "hidden",
        "#secret",
        "credential",
        "bob",
        "do-not-display",
        "unexpected_secret",
    ):
        assert secret not in output
    for key in (
        "profile_name",
        "repo_root",
        "db_path",
        "inbox_root",
        "inbox_raw_dir",
        "chatgpt_inbox_dir",
        "output_root",
        "qdrant_collection",
        "embedding_model",
        "prompts_dir",
        "samples_dir",
    ):
        assert str(config[key]) in output
    assert "http://localhost:11434/" in output
    assert "Qdrant host: localhost" in output


def test_read_only_resolution_does_not_touch_database_or_services(profile_repo: Path) -> None:
    database = profile_repo / "sentinel.db"
    database.write_bytes(b"synthetic sentinel")
    (profile_repo / ".env").write_text("JARVIS_DB_PATH=sentinel.db\n", encoding="utf-8")
    before = database.read_bytes(), database.stat().st_mtime_ns
    with (
        patch.object(Path, "mkdir", side_effect=AssertionError("Unexpected directory creation")),
        patch.object(sqlite3, "connect", side_effect=AssertionError("Unexpected SQLite access")),
        patch("socket.socket.connect", side_effect=AssertionError("Unexpected network access")),
    ):
        assert "sentinel.db" in format_config(load_config())
    assert (database.read_bytes(), database.stat().st_mtime_ns) == before


@pytest.mark.parametrize("section", ["model", "outputs", "memory", "paths", "ollama"])
def test_invalid_default_sections(profile_repo: Path, section: str) -> None:
    (profile_repo / "config.yaml").write_text(
        yaml.safe_dump({"defaults": {section: []}}), encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match=section):
        load_config()


@pytest.mark.parametrize("value", [None, [], 42])
def test_invalid_shared_string_setting(profile_repo: Path, value: Any) -> None:
    (profile_repo / "config.yaml").write_text(
        yaml.safe_dump({"defaults": {"ollama": {"base_url": value}}}), encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="ollama_base_url"):
        load_config()


def test_environment_interpolation_preserves_process_precedence(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (profile_repo / ".env").write_text(
        "BASE=model-from-file\nLOCAL_MODEL_NAME=${BASE}\n", encoding="utf-8"
    )
    monkeypatch.setenv("BASE", "model-from-environment")
    before = dict(os.environ)
    assert load_config()["local_model_name"] == "model-from-environment"
    assert dict(os.environ) == before


def test_malformed_environment_fails_without_echoing_secrets(profile_repo: Path) -> None:
    (profile_repo / ".env").write_text("JARVIS_PROFILE='unterminated-secret\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid .env") as error:
        load_config()
    assert "unterminated-secret" not in str(error.value)


@pytest.mark.parametrize(
    ("contents", "expected_local", "expected_openai"),
    [
        (
            "base: &base {local: inherited, openai: inherited-openai}\n"
            "defaults:\n  model:\n    <<: *base\n    local: explicit\n",
            "explicit",
            "inherited-openai",
        ),
        (
            "first: &first {local: first}\n"
            "second: &second {local: second, openai: second-openai}\n"
            "defaults:\n  model:\n    <<: [*first, *second]\n",
            "first",
            "second-openai",
        ),
        (
            "base: &base {local: inherited, openai: inherited-openai}\n"
            "override: &override {<<: *base, local: override}\n"
            "defaults: {model: *override}\n",
            "override",
            "inherited-openai",
        ),
        (
            "base: &base {local: inherited, openai: inherited-openai}\n"
            "defaults: {model: {local: explicit, <<: *base}}\n",
            "explicit",
            "inherited-openai",
        ),
    ],
)
def test_yaml_merges_preserve_standard_precedence(
    profile_repo: Path, contents: str, expected_local: str, expected_openai: str
) -> None:
    (profile_repo / "config.yaml").write_text(contents, encoding="utf-8")
    resolved = load_config()
    assert resolved["local_model_name"] == expected_local
    assert resolved["openai_model"] == expected_openai
    assert not (profile_repo / "data").exists()


def test_profile_yaml_merge_allows_explicit_storage_overrides(profile_repo: Path) -> None:
    (profile_repo / "profiles.yaml").write_text(
        "profiles:\n"
        "  personal: &personal\n"
        "    db_path: data/jarvis.db\n"
        "    qdrant_collection: jarvis_fragments\n"
        "    inbox_root: inbox\n"
        "    output_root: OUTPUTS\n"
        "    user_name: Personal owner\n"
        "  mom:\n"
        "    <<: *personal\n"
        "    db_path: data/profiles/mom/jarvis.db\n"
        "    qdrant_collection: jarvis_mom_fragments\n"
        "    inbox_root: data/profiles/mom/inbox\n"
        "    output_root: data/profiles/mom/OUTPUTS\n"
        "    user_name: Mom\n",
        encoding="utf-8",
    )
    resolved = load_config("mom")
    assert resolved["db_path"] == str(profile_repo / "data/profiles/mom/jarvis.db")
    assert resolved["qdrant_collection"] == "jarvis_mom_fragments"
    assert resolved["user_name"] == "Mom"
    assert not (profile_repo / "data").exists()


@pytest.mark.parametrize(
    "contents",
    [
        "defaults: {model: {local: first, local: second}}\n",
        "base: &base {local: first, local: second}\ndefaults: {model: {<<: *base}}\n",
        "base: &base {local: inherited}\n"
        "defaults: {model: {<<: *base, local: first, local: second}}\n",
        "base: &base {local: inherited}\n" "defaults: {model: {<<: *base, <<: *base}}\n",
        "base: &base {local: inherited}\n"
        "defaults: {model: {<<: *base, nested: {field: first, field: second}}}\n",
    ],
)
def test_explicit_yaml_duplicates_are_rejected_even_with_merges(
    profile_repo: Path, contents: str
) -> None:
    (profile_repo / "config.yaml").write_text(contents, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unique string keys"):
        load_config()
    assert not (profile_repo / "data").exists()
