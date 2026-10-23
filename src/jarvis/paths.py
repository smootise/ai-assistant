"""Profile-owned managed paths and read-only external-source boundaries."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from jarvis.profiles import ConfigurationError, ProfileContext


class PathBoundaryError(ConfigurationError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance(path: Path) -> dict[str, str]:
    return {"original_path": str(path.resolve()), "sha256": sha256_file(path)}


def identifier(value: Any, label: str = "identifier") -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or value.endswith((".", " "))
        or any(ord(char) < 32 or char in '/\\<>:"|?*' for char in value)
        or re.match(r"(?i)^(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", value)
    ):
        raise PathBoundaryError(f"Unsafe {label}: {value!r}")
    return value


def _absolute(value: str | Path) -> Path:
    text = str(value)
    if not text or "\x00" in text or ".." in text.replace("\\", "/").split("/"):
        raise PathBoundaryError(f"Unsafe path: {text!r}")
    windows = PureWindowsPath(text)
    if windows.drive and not windows.is_absolute():
        raise PathBoundaryError(f"Drive-relative path is not allowed: {text}")
    if os.name == "nt" and windows.root and not windows.drive:
        raise PathBoundaryError(f"Rooted paths must include a drive: {text}")
    if os.name == "nt":
        for part in Path(text).parts:
            if part != Path(text).anchor:
                identifier(part, "path component")
    try:
        return Path(text).absolute()
    except (OSError, ValueError) as exc:
        raise PathBoundaryError(f"Cannot resolve path: {text}") from exc


def _within(root: Path, candidate: Path) -> bool:
    return candidate.is_relative_to(root)


def confined(root: Path, value: str | Path) -> Path:
    candidate = _absolute(value)
    try:
        resolved_root = root.resolve()
        if resolved_root != root.absolute():
            raise PathBoundaryError(f"Managed root was redirected: {root}")
        resolved = candidate.resolve()
        if not _within(resolved_root, resolved):
            raise PathBoundaryError(f"Path is outside the active managed directory: {candidate}")
        if resolved.is_file() and resolved.stat().st_nlink > 1:
            raise PathBoundaryError(f"Managed hard-linked files are not supported: {candidate}")
        return resolved
    except (OSError, RuntimeError) as exc:
        raise PathBoundaryError(f"Cannot validate managed path: {candidate}") from exc


def validate_record(data: Any, conversation_id: str | None = None) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if key in {"conversation_id", "parent_conversation_id", "segment_id"}:
                identifier(value, key)
                if key != "segment_id" and conversation_id is not None and value != conversation_id:
                    raise PathBoundaryError(
                        f"Artifact {key} does not match its conversation directory"
                    )
            if key in {"segment_index", "fragment_index"} and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise PathBoundaryError(f"Invalid artifact {key}")
            validate_record(value, conversation_id)
    elif isinstance(data, list):
        for value in data:
            validate_record(value, conversation_id)


def validate_tree(
    root: Path, value: Path, conversation_id: str | None = None, *, repairable: bool = False
) -> None:
    pending = [value]
    visited: set[Path] = set()
    while pending:
        path = confined(root, pending.pop())
        if path in visited:
            continue
        visited.add(path)
        if path.is_dir():
            pending.extend(path.iterdir())
        elif path.is_file() and path.suffix == ".json":
            try:
                with path.open(encoding="utf-8") as stream:
                    data = json.load(stream)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if repairable:
                    continue
                raise
            validate_record(data, conversation_id)


@dataclass(frozen=True)
class ProfilePaths:
    context: ProfileContext
    profiles: Mapping[str, ProfileContext]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ProfilePaths":
        context = config.get("profile_context")
        if context is not None:
            for key, expected in (
                ("db_path", context.db_path),
                ("inbox_root", context.inbox_root),
                ("output_root", context.output_root),
                ("chatgpt_inbox_dir", context.chatgpt_inbox_dir),
            ):
                if key in config and Path(config[key]).resolve() != expected:
                    raise PathBoundaryError(
                        f"Configuration '{key}' disagrees with selected profile"
                    )
            for key, expected in (
                ("profile_name", context.name),
                ("qdrant_collection", context.qdrant_collection),
            ):
                if key in config and config[key] != expected:
                    raise PathBoundaryError(
                        f"Configuration '{key}' disagrees with selected profile"
                    )
        else:
            if "profile_name" in config or "profiles" in config:
                raise PathBoundaryError("Named profile routing requires a resolved profile context")
            repo = Path(config.get("repo_root", ".")).resolve()
            context = ProfileContext(
                name=config.get("profile_name", "personal"),
                db_path=Path(config.get("db_path", repo / "data/jarvis.db")).resolve(),
                qdrant_collection=config.get("qdrant_collection", "jarvis_fragments"),
                inbox_root=Path(config.get("inbox_root", repo / "inbox")).resolve(),
                output_root=Path(config.get("output_root", repo / "OUTPUTS")).resolve(),
            )
        return cls(context, config.get("profiles", {context.name: context}))

    def inbox(self, override: str | Path | None = None) -> Path:
        return confined(
            self.context.inbox_root,
            override if override is not None else self.context.chatgpt_inbox_dir,
        )

    def output(self, *parts: str) -> Path:
        for part in parts:
            identifier(part)
        return confined(self.context.output_root, self.context.output_root.joinpath(*parts))

    def conversation(self, conversation_id: str, inbox: str | Path | None = None) -> Path:
        identifier(conversation_id, "conversation_id")
        return confined(self.context.inbox_root, self.inbox(inbox) / conversation_id)

    def source(self, value: str | Path) -> Path:
        candidate = _absolute(value)
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:
            raise PathBoundaryError(f"Cannot validate source path: {candidate}") from exc
        for profile in self.profiles.values():
            for suffix in ("", "-wal", "-shm", "-journal"):
                database = Path(f"{profile.db_path}{suffix}").resolve()
                if resolved == database:
                    raise PathBoundaryError(f"Database files cannot be source inputs: {candidate}")
            for root in (profile.inbox_root, profile.output_root):
                lexical = _within(root, candidate)
                actual = _within(root.resolve(), resolved)
                if profile.name != self.context.name and (lexical or actual):
                    raise PathBoundaryError(f"Source belongs to another profile: {candidate}")
                if profile.name == self.context.name and lexical:
                    confined(root, candidate)
        if resolved.is_file() and resolved.stat().st_nlink > 1:
            raise PathBoundaryError(f"Hard-linked sources are not supported: {candidate}")
        return resolved
