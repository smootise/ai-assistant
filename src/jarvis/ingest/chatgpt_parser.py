"""ChatGPT conversation export parser.

Reconstructs the exact visible conversation branch from a raw ChatGPT
export JSON using the `current_node` field and backward parent-walk.
Filters to only visible user/assistant text messages.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Roles that are kept after filtering.
_VISIBLE_ROLES = {"user", "assistant"}

# Recipient values that indicate hidden memory/bio messages.
_HIDDEN_RECIPIENTS = {"bio"}


def load_raw_export(path: str) -> Dict[str, Any]:
    """Load a raw ChatGPT conversation export JSON file.

    Args:
        path: Path to the raw export JSON file.

    Returns:
        Parsed export dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raw export not found: {path}")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e


def reconstruct_active_path(mapping: Dict[str, Any], current_node: Optional[str]) -> List[str]:
    """Walk backward from current_node to root, then reverse.

    This gives the exact sequence of node IDs that forms the visible
    conversation branch as the user last saw it.

    Falls back to following the last child at each branch if current_node
    is not provided.

    Args:
        mapping: The raw `mapping` dict from the export.
        current_node: The `current_node` value from the export top level.

    Returns:
        Ordered list of node IDs from root to current_node.
    """
    if current_node and current_node in mapping:
        path: List[str] = []
        node_id: Optional[str] = current_node
        while node_id:
            path.append(node_id)
            node_id = mapping[node_id].get("parent")
            if node_id and node_id not in mapping:
                logger.warning(f"Parent node {node_id!r} not found in mapping, stopping walk")
                break
        path.reverse()
        logger.debug(f"Active path: {len(path)} nodes via current_node backward walk")
        return path

    # Fallback: find root and follow last child at each step
    logger.warning("current_node missing or not in mapping — falling back to last-child walk")
    root = next((k for k, v in mapping.items() if v.get("parent") is None), None)
    if not root:
        return []
    path = []
    node_id = root
    visited: set = set()
    while node_id:
        if node_id not in mapping:
            logger.warning(f"Node {node_id!r} not found in mapping during last-child walk, stopping")  # noqa: E501
            break
        if node_id in visited:
            logger.warning(f"Cycle detected at node {node_id!r}, stopping walk")
            break
        visited.add(node_id)
        path.append(node_id)
        children = mapping[node_id].get("children", [])
        # Filter to children actually present in mapping
        valid_children = [c for c in children if c in mapping]
        node_id = valid_children[-1] if valid_children else None
    return path


def _extract_text(content: Dict[str, Any]) -> str:
    """Join content.parts into a single clean text string.

    Args:
        content: The `content` dict from a message node.

    Returns:
        Stripped text string, or empty string if nothing visible.
    """
    parts = content.get("parts", [])
    chunks = []
    for part in parts:
        if isinstance(part, str):
            chunks.append(part)
        elif isinstance(part, dict):
            # Some parts are dicts (e.g. image refs) — skip them
            pass
    return "\n".join(chunks).strip()


def _is_visible(msg: Dict[str, Any]) -> bool:
    """Return True if a message should be included in the normalized output.

    Exclusion rules (any one is sufficient to drop):
    - author.role not in {user, assistant}
    - recipient == "bio" (memory-update assistant messages)
    - content.content_type != "text"
    - content text is empty after joining parts
    - metadata.is_visually_hidden_from_conversation is True

    Args:
        msg: A raw message dict.

    Returns:
        True if the message is visible and should be kept.
    """
    role = msg.get("author", {}).get("role", "")
    if role not in _VISIBLE_ROLES:
        return False

    if msg.get("recipient") in _HIDDEN_RECIPIENTS:
        return False

    content = msg.get("content", {})
    if content.get("content_type") != "text":
        return False

    if msg.get("metadata", {}).get("is_visually_hidden_from_conversation"):
        return False

    text = _extract_text(content)
    if not text:
        return False

    return True


def _make_message_id(
    msg: Dict[str, Any],
    conversation_id: str,
    position: int,
) -> str:
    """Return a stable message ID.

    Primary: raw message.id when present.
    Fallback: SHA-256 of (conversation_id, speaker, content, created_at).

    Args:
        msg: Raw message dict.
        conversation_id: Parent conversation ID for fallback hashing.
        position: Ordinal position used only for logging.

    Returns:
        Stable string ID.
    """
    raw_id = msg.get("id", "").strip()
    if raw_id:
        return raw_id

    role = msg.get("author", {}).get("role", "")
    content = _extract_text(msg.get("content", {}))
    created_at = str(msg.get("create_time") or "")
    payload = f"{conversation_id}|{role}|{content}|{created_at}"
    hashed = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    logger.debug(f"Position {position}: no raw message ID, using hash {hashed}")
    return f"hash_{hashed}"


def _to_iso(ts: Optional[float]) -> Optional[str]:
    """Convert a Unix float timestamp to ISO-8601 UTC string, or None."""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _collapse_adjacent_user_retries(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop earlier of two adjacent identical user messages (retry dedup).

    Normalizes only whitespace/line-endings for comparison.
    Keeps the later message. Does not apply fuzzy matching.

    Args:
        messages: Ordered list of visible message dicts (pre-normalized).

    Returns:
        List with retry duplicates removed.
    """
    if len(messages) < 2:
        return messages

    result = []
    i = 0
    while i < len(messages):
        current = messages[i]
        if (
            i + 1 < len(messages)
            and current["author"]["role"] == "user"
            and messages[i + 1]["author"]["role"] == "user"
        ):
            c1 = " ".join(current["_raw_text"].split())
            c2 = " ".join(messages[i + 1]["_raw_text"].split())
            if c1 == c2:
                logger.debug(
                    f"Collapsing adjacent duplicate user retry at position {i}"
                )
                i += 1  # skip the earlier, keep the next
                continue
        result.append(current)
        i += 1
    return result


def parse_export(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a raw ChatGPT export into an ordered list of normalized messages.

    Steps:
    1. Reconstruct active path via current_node backward walk.
    2. Filter to visible user/assistant text messages.
    3. Collapse adjacent identical user messages (retry dedup).
    4. Assign position from final ordered sequence.

    Args:
        raw: Parsed raw export dict.

    Returns:
        List of normalized message dicts with keys:
            message_id, speaker, created_at, position, content.
    """
    mapping: Dict[str, Any] = raw.get("mapping", {})
    current_node: Optional[str] = raw.get("current_node")
    conversation_id: str = raw.get("conversation_id", "")

    active_path = reconstruct_active_path(mapping, current_node)
    logger.info(f"Active path: {len(active_path)} nodes")

    # Collect raw visible messages in path order
    raw_visible: List[Dict[str, Any]] = []
    for node_id in active_path:
        node = mapping.get(node_id, {})
        msg = node.get("message")
        if not msg:
            continue
        if not _is_visible(msg):
            continue
        # Attach raw_text for retry-dedup comparison
        msg["_raw_text"] = _extract_text(msg.get("content", {}))
        raw_visible.append(msg)

    logger.info(f"Visible messages before retry-dedup: {len(raw_visible)}")
    raw_visible = _collapse_adjacent_user_retries(raw_visible)
    logger.info(f"Visible messages after retry-dedup: {len(raw_visible)}")

    # Build normalized message dicts
    normalized: List[Dict[str, Any]] = []
    for position, msg in enumerate(raw_visible):
        message_id = _make_message_id(msg, conversation_id, position)
        normalized.append(
            {
                "message_id": message_id,
                "speaker": msg["author"]["role"],
                "created_at": _to_iso(msg.get("create_time")),
                "position": position,
                "content": msg["_raw_text"],
            }
        )

    return normalized
