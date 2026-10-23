"""Atomic artifacts and stage completion checks shared by processing and recovery."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from jarvis.paths import confined, validate_record


def atomic_text(path: Path, text: str) -> None:
    path = confined(path.parent.resolve(), path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, data: Any) -> None:
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def valid_output(data: Any, kind: str, expected: dict[str, Any] | None = None) -> bool:
    if not isinstance(data, dict) or data.get("status") not in {"ok", "degraded"}:
        return False
    if data.get("source_kind") != kind:
        return False
    if expected and any(data.get(key) != value for key, value in expected.items()):
        return False
    validate_record(data)
    if kind in {"conversation", "ai_chat_segment", "ai_chat_topic"}:
        confidence = data.get("confidence", 0.0)
        return (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0 <= confidence <= 1
            and isinstance(data.get("summary"), str)
            and bool(data["summary"].strip())
            and all(
                isinstance(data.get(key, []), list)
                and all(isinstance(item, str) for item in data.get(key, []))
                for key in ("bullets", "action_items")
            )
        )
    statements = data.get("statements")
    if not isinstance(statements, list) or any(
        not isinstance(item, dict)
        or item.get("speaker") not in {"user", "assistant"}
        or not isinstance(item.get("text"), str)
        for item in statements
    ):
        return False
    if kind == "ai_chat_extract":
        indices = [item.get("statement_index") for item in statements]
        return all(type(index) is int for index in indices) and indices == list(
            range(len(statements))
        )
    return (
        isinstance(data.get("text"), str)
        and bool(data["text"].strip())
        and isinstance(data.get("title"), str)
    )


def load_output(
    path: Path, kind: str, expected: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if valid_output(data, kind, expected) else None


def fragment_outputs(directory: Path, extract: dict[str, Any]) -> list[dict[str, Any]] | None:
    expected = {
        "segment_id": extract["segment_id"],
        "segment_index": extract["segment_index"],
        "parent_conversation_id": extract["parent_conversation_id"],
    }
    manifest = directory / "fragment_manifest.json"
    if manifest.exists():
        try:
            documents = json.loads(manifest.read_text(encoding="utf-8"))["outputs"]
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
            return None
    else:
        files = sorted(directory.glob("fragment_*.json"))
        documents = [load_output(path, "ai_chat_fragment", expected) for path in files]
        # Legacy files have no completion marker. Require full statement coverage.
        covered = [
            (statement.get("speaker"), statement.get("text"))
            for document in documents
            if document
            for statement in document.get("statements", [])
        ]
        if not documents or any(
            (statement["speaker"], statement["text"]) not in covered
            for statement in extract.get("statements", [])
        ):
            return None
    if (
        not isinstance(documents, list)
        or not documents
        or any(
            not valid_output(document, "ai_chat_fragment", {**expected, "fragment_index": index})
            for index, document in enumerate(documents)
        )
    ):
        return None
    return documents
