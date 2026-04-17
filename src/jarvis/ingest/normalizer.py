"""Normalizer: builds and merges normalized conversation JSON files.

Handles incremental re-import by deduplicating on message_id and
merging new messages into the existing normalized file while preserving
the canonical traversal order.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def build_normalized(
    raw: Dict[str, Any],
    messages: List[Dict[str, Any]],
    raw_export_path: str,
) -> Dict[str, Any]:
    """Build a fresh normalized conversation document.

    Args:
        raw: Parsed raw export dict (for top-level metadata).
        messages: Ordered list of normalized message dicts from the parser.
        raw_export_path: Path to the raw export file (stored as provenance).

    Returns:
        Normalized conversation dict.
    """
    conversation_id = raw.get("conversation_id", Path(raw_export_path).stem)
    return {
        "conversation_id": conversation_id,
        "title": raw.get("title", ""),
        "source_platform": "chatgpt",
        "source_file": Path(raw_export_path).name,
        "raw_export_path": str(raw_export_path),
        "source_url": None,
        "created_at": _to_iso(raw.get("create_time")),
        "updated_at": _to_iso(raw.get("update_time")),
        "imported_at": _now_iso(),
        "message_count": len(messages),
        "messages": messages,
    }


def merge_normalized(
    existing: Dict[str, Any],
    new_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge new messages into an existing normalized conversation.

    Deduplication: any message whose message_id already exists in the
    existing document is skipped. New messages are appended after
    existing ones, then positions are reassigned by index.

    The traversal order from the new parse takes precedence for new
    messages; existing messages keep their relative order.

    Args:
        existing: Previously saved normalized conversation dict.
        new_messages: Messages from a fresh parse of the same conversation.

    Returns:
        Updated normalized conversation dict.
    """
    known_ids = {m["message_id"] for m in existing.get("messages", [])}
    appended = 0
    merged = list(existing.get("messages", []))

    for msg in new_messages:
        if msg["message_id"] not in known_ids:
            merged.append(msg)
            known_ids.add(msg["message_id"])
            appended += 1

    # Reassign positions from final merged order
    for i, msg in enumerate(merged):
        msg["position"] = i

    logger.info(
        f"Merge complete: {len(existing.get('messages', []))} existing + "
        f"{appended} new = {len(merged)} total messages"
    )

    updated = dict(existing)
    updated["messages"] = merged
    updated["message_count"] = len(merged)
    updated["imported_at"] = _now_iso()
    return updated


def load_normalized(path: Path) -> Optional[Dict[str, Any]]:
    """Load an existing normalized file, or return None if it doesn't exist.

    Args:
        path: Path to the normalized JSON file.

    Returns:
        Parsed dict, or None.
    """
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_normalized(normalized: Dict[str, Any], path: Path) -> None:
    """Write a normalized conversation to disk atomically.

    Writes to a .tmp file then renames to avoid partial writes.

    Args:
        normalized: Normalized conversation dict.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    logger.info(f"Normalized conversation saved to {path}")
