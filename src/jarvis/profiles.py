"""Read-only profile resolution and managed-storage validation."""

import os
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ProfileContext:
    name: str
    db_path: Path
    qdrant_collection: str
    inbox_root: Path
    output_root: Path
    user_name: str = ""

    @property
    def chatgpt_inbox_dir(self) -> Path:
        return self.inbox_root / "ai_chat" / "chatgpt"

    @property
    def inbox_raw_dir(self) -> Path:
        return self.chatgpt_inbox_dir / "raw"


_PROFILE_NAME = re.compile(r"[a-z][a-z0-9_-]*\Z")
_COLLECTION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_STORAGE_FIELDS = {"db_path", "qdrant_collection", "inbox_root", "output_root"}
_PROFILE_FIELDS = _STORAGE_FIELDS | {"user_name"}


def resolve_path(value: Any, repo_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigurationError(f"{label} must be a non-empty path string")
    try:
        candidate = Path(value).expanduser()
        if os.name == "nt" and candidate.drive and not candidate.is_absolute():
            raise ConfigurationError(f"{label} must not be drive-relative: {value}")
        if os.name == "nt" and candidate.root and not candidate.drive:
            raise ConfigurationError(f"{label} must include a drive: {value}")
        if os.name == "nt":
            for part in candidate.parts:
                if part in {candidate.anchor, ".."}:
                    continue
                if (
                    part.endswith((" ", "."))
                    or any(char in part for char in '<>:"|?*')
                    or re.match(r"(?i)^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", part)
                ):
                    raise ConfigurationError(f"{label} contains an invalid Windows path component")
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        return candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Cannot resolve {label}") from exc


def _same_path(first: Path, second: Path) -> bool:
    if os.path.normcase(str(first)) == os.path.normcase(str(second)):
        return True
    if first.exists() and second.exists():
        return first.samefile(second)
    return False


def _contains(root: Path, candidate: Path) -> bool:
    root_text = os.path.normcase(str(root))
    candidate_text = os.path.normcase(str(candidate))
    try:
        return os.path.commonpath((root_text, candidate_text)) == root_text
    except ValueError:
        return False


def _database_files(profile: ProfileContext) -> tuple[Path, ...]:
    return tuple(
        Path(f"{profile.db_path}{suffix}").resolve() for suffix in ("", "-wal", "-shm", "-journal")
    )


def _conflict(first: ProfileContext, left: Path, second: ProfileContext, right: Path) -> None:
    raise ConfigurationError(
        f"Storage conflict: profile '{first.name}' ({left}) and "
        f"profile '{second.name}' ({right})"
    )


def _overlaps(first: Path, second: Path) -> bool:
    return _contains(first, second) or _contains(second, first) or _same_path(first, second)


def _validate_storage(profiles: Mapping[str, ProfileContext]) -> None:
    database_files = {name: _database_files(profile) for name, profile in profiles.items()}
    for profile in profiles.values():
        reserved_files = database_files[profile.name]
        for path in (*reserved_files, profile.inbox_root, profile.output_root):
            for parent in path.parents:
                if parent.exists() and not parent.is_dir():
                    raise ConfigurationError(
                        f"Profile '{profile.name}' path has a non-directory parent: {path}"
                    )
        for db_file in reserved_files:
            if db_file.exists() and not db_file.is_file():
                raise ConfigurationError(
                    f"Profile '{profile.name}' database or sidecar is not a file: {db_file}"
                )
        for left, right in combinations(reserved_files, 2):
            if _overlaps(left, right):
                _conflict(profile, left, profile, right)
        for root in (profile.inbox_root, profile.output_root):
            if root.exists() and not root.is_dir():
                raise ConfigurationError(
                    f"Profile '{profile.name}' storage root is not a directory: {root}"
                )
            for db_file in reserved_files:
                if _overlaps(root, db_file):
                    _conflict(profile, root, profile, db_file)
        if _overlaps(profile.inbox_root, profile.output_root):
            _conflict(profile, profile.inbox_root, profile, profile.output_root)

    for first, second in combinations(profiles.values(), 2):
        if first.qdrant_collection == second.qdrant_collection:
            raise ConfigurationError(
                f"Profiles '{first.name}' and '{second.name}' share Qdrant collection "
                f"'{first.qdrant_collection}'"
            )
        for left in database_files[first.name]:
            for right in database_files[second.name]:
                if _overlaps(left, right):
                    _conflict(first, left, second, right)
        for left in (first.inbox_root, first.output_root):
            for right in (second.inbox_root, second.output_root):
                if _overlaps(left, right):
                    _conflict(first, left, second, right)
        for owner, other in ((first, second), (second, first)):
            for root in (other.inbox_root, other.output_root):
                for db_file in database_files[owner.name]:
                    if _overlaps(root, db_file):
                        _conflict(owner, db_file, other, root)


def resolve_profiles(
    registry: Mapping[str, Any],
    repo_root: Path,
    personal_defaults: Mapping[str, Any],
    environment: Mapping[str, str],
) -> dict[str, ProfileContext]:
    if set(registry) != {"profiles"} or not isinstance(registry["profiles"], dict):
        raise ConfigurationError("profiles.yaml must contain a 'profiles' mapping only")
    entries = registry["profiles"]
    if "personal" not in entries:
        raise ConfigurationError("profiles.yaml must register the default 'personal' profile")

    profiles: dict[str, ProfileContext] = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
            raise ConfigurationError("Profile names must match [a-z][a-z0-9_-]*")
        if not isinstance(entry, dict) or set(entry) - _PROFILE_FIELDS:
            raise ConfigurationError(f"Profile '{name}' has invalid configuration fields")
        values = dict(entry)
        if name == "personal":
            values = {**personal_defaults, **values}
            for variable, field in (
                ("JARVIS_DB_PATH", "db_path"),
                ("JARVIS_OUTPUT_ROOT", "output_root"),
                ("JARVIS_USER_NAME", "user_name"),
            ):
                if variable in environment:
                    values[field] = environment[variable]
        elif _STORAGE_FIELDS - values.keys():
            missing = ", ".join(sorted(_STORAGE_FIELDS - values.keys()))
            raise ConfigurationError(f"Profile '{name}' is missing: {missing}")

        collection = values.get("qdrant_collection")
        if not isinstance(collection, str) or not _COLLECTION_NAME.fullmatch(collection):
            raise ConfigurationError(f"Profile '{name}' has an invalid Qdrant collection name")
        user_name = values.get("user_name", "")
        if not isinstance(user_name, str):
            raise ConfigurationError(f"Profile '{name}' user_name must be a string")
        profiles[name] = ProfileContext(
            name=name,
            db_path=resolve_path(values.get("db_path"), repo_root, f"{name}.db_path"),
            qdrant_collection=collection,
            inbox_root=resolve_path(values.get("inbox_root"), repo_root, f"{name}.inbox_root"),
            output_root=resolve_path(values.get("output_root"), repo_root, f"{name}.output_root"),
            user_name=user_name,
        )
    try:
        _validate_storage(profiles)
    except ConfigurationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigurationError("Cannot inspect profile storage paths") from exc
    return profiles


def require_personal_runtime(config: Mapping[str, Any]) -> None:
    if config.get("profile_name", "personal") != "personal":
        raise ConfigurationError(
            "Phase 2 supports non-personal profiles only for 'config show'; "
            "processing and serving require profile 'personal'"
        )
    # Pipeline and web isolation remain pending; retain the rollout boundary.
    if config.get("qdrant_collection", "jarvis_fragments") != "jarvis_fragments":
        raise ConfigurationError("Custom personal collections are inspection-only in Phase 2")
    if "inbox_root" in config:
        expected = (Path(config["repo_root"]) / "inbox").resolve()
        if not _same_path(Path(config["inbox_root"]), expected):
            raise ConfigurationError("Custom personal inboxes are inspection-only in Phase 2")
