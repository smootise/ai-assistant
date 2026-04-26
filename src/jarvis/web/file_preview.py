"""ID-first whitelisted raw file reader for the JARVIS web UI.

Files are resolved from entity IDs stored in SQLite — no user-supplied paths.
Every resolved path is re-validated against an approved root whitelist before reading.
"""

import json
from pathlib import Path
from typing import Tuple

from jarvis.store import SummaryStore

_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
_ALLOWED_EXTENSIONS = {".json", ".md", ".txt"}


class PreviewError(Exception):
    """Raised when a preview cannot be served safely."""


def _resolve_and_validate(
    candidate: Path, *allowed_roots: Path
) -> Path:
    """Resolve candidate and assert it sits under one of the allowed roots.

    Raises PreviewError on traversal, symlink escape, bad extension, or missing file.
    """
    if candidate.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise PreviewError(f"File type not previewable: {candidate.suffix}")

    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, FileNotFoundError):
        raise PreviewError("File not found on disk.")

    # Guard against symlink escapes — check the real path is under an approved root
    for root in allowed_roots:
        root_resolved = root.resolve()
        try:
            resolved.relative_to(root_resolved)
            return resolved
        except ValueError:
            continue

    raise PreviewError("File is outside the approved preview roots.")


def _read_file(path: Path) -> Tuple[str, str]:
    """Read a whitelisted file and return (content, kind).

    kind is one of "json", "markdown", "text".
    """
    size = path.stat().st_size
    if size > _MAX_BYTES:
        raise PreviewError(f"File too large to preview ({size:,} bytes; limit {_MAX_BYTES:,}).")

    raw = path.read_text(encoding="utf-8", errors="replace")

    ext = path.suffix.lower()
    if ext == ".json":
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, indent=2, ensure_ascii=False), "json"
        except json.JSONDecodeError:
            return raw, "text"
    if ext == ".md":
        return raw, "markdown"
    return raw, "text"


def _roots(output_root: Path, inbox_root: Path) -> Tuple[Path, Path]:
    return output_root, inbox_root


def read_for_source(
    store: SummaryStore,
    source_id: str,
    output_root: Path,
    inbox_root: Path,
) -> Tuple[str, str, str]:
    """Return (content, kind, filename) for a source file, or raise PreviewError."""
    source = store.get_source_file(source_id)
    if source is None:
        raise PreviewError("Source file not found in database.")

    storage_path = source.get("storage_path", "")
    if not storage_path:
        raise PreviewError("No storage path recorded for this source file.")

    candidate = Path(storage_path)
    if not candidate.is_absolute():
        # Try resolving relative to inbox and output roots
        for root in (inbox_root, output_root):
            attempt = root / storage_path
            if attempt.exists():
                candidate = attempt
                break

    validated = _resolve_and_validate(candidate, output_root, inbox_root)
    content, kind = _read_file(validated)
    return content, kind, validated.name


def read_for_segment(
    store: SummaryStore,
    segment_id: str,
    inbox_root: Path,
) -> Tuple[str, str, str]:
    """Resolve segment_NNN.json from inbox and return (content, kind, filename)."""
    segment = store.get_segment(segment_id)
    if segment is None:
        raise PreviewError("Segment not found in database.")

    conv_id = segment["conversation_id"]
    seg_idx = segment["segment_index"]
    filename = f"segment_{seg_idx:03d}.json"

    # ChatGPT layout: inbox/ai_chat/chatgpt/<conv_id>/segments/<filename>
    candidate = inbox_root / "ai_chat" / "chatgpt" / conv_id / "segments" / filename

    validated = _resolve_and_validate(candidate, inbox_root)
    content, kind = _read_file(validated)
    return content, kind, filename


def read_for_extract(
    store: SummaryStore,
    extract_id: str,
    output_root: Path,
) -> Tuple[str, str, str]:
    """Resolve extract_NNN.json from OUTPUTS and return (content, kind, filename)."""
    extract = store.get_extract(extract_id)
    if extract is None:
        raise PreviewError("Extract not found in database.")

    conv_id = extract["parent_conversation_id"]
    seg_idx = extract["segment_index"]
    filename = f"extract_{seg_idx:03d}.json"
    candidate = output_root / conv_id / "extracts" / filename

    validated = _resolve_and_validate(candidate, output_root)
    content, kind = _read_file(validated)
    return content, kind, filename


def read_for_fragment(
    store: SummaryStore,
    fragment_id: str,
    output_root: Path,
) -> Tuple[str, str, str]:
    """Resolve fragment_NNN.json from OUTPUTS and return (content, kind, filename)."""
    fragment = store.get_fragment(fragment_id)
    if fragment is None:
        raise PreviewError("Fragment not found in database.")

    extract_id = fragment["extract_id"]
    extract = store.get_extract(extract_id)
    if extract is None:
        raise PreviewError("Parent extract not found in database.")

    conv_id = extract["parent_conversation_id"]
    seg_idx = extract["segment_index"]
    frag_idx = fragment["fragment_index"]
    filename = f"fragment_{frag_idx:03d}.json"
    candidate = output_root / conv_id / "fragments" / f"segment_{seg_idx:03d}" / filename

    validated = _resolve_and_validate(candidate, output_root)
    content, kind = _read_file(validated)
    return content, kind, filename
