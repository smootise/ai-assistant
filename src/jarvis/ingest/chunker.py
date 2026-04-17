"""Chunker for normalized AI chat conversations.

Implements the source-specific `ai_chat` chunking strategy:
  user -> assistant -> user (with overlap on the trailing user message)

Chunk i:   [user_n, assistant_n, user_{n+1}]
Chunk i+1: [user_{n+1}, assistant_{n+1}, user_{n+2}]

Special cases:
- If the conversation ends after an assistant reply, emit a final
  2-message chunk (user, assistant).
- A trailing unmatched user message is NOT emitted as a normal chunk.
  It is recorded as a pending_tail in the manifest.
- If the ideal pattern breaks (consecutive same-role messages), fall
  back gracefully: emit whatever contiguous messages are available
  without dropping any, and advance the cursor.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _format_chunk_text(messages: List[Dict[str, Any]]) -> str:
    """Format a list of normalized messages as a readable transcript.

    Args:
        messages: Ordered normalized message dicts.

    Returns:
        String with each turn on separate lines, e.g.:
            user: ...
            assistant: ...
    """
    parts = []
    for msg in messages:
        speaker = msg["speaker"]
        content = msg["content"].strip()
        parts.append(f"{speaker}: {content}")
    return "\n\n".join(parts)


def chunk_conversation(
    normalized: Dict[str, Any],
) -> Dict[str, Any]:
    """Chunk a normalized conversation into summarization-ready chunks.

    Args:
        normalized: Normalized conversation dict (from normalizer.py).

    Returns:
        Dict with keys:
            chunks: list of chunk dicts
            pending_tail: optional single message dict (trailing unmatched user)
            manifest_meta: top-level metadata for the manifest
    """
    messages = normalized.get("messages", [])
    conversation_id = normalized.get("conversation_id", "")
    chunks: List[Dict[str, Any]] = []
    pending_tail: Optional[Dict[str, Any]] = None

    i = 0
    chunk_index = 0

    while i < len(messages):
        current = messages[i]
        role = current["speaker"]

        # --- Ideal path: user -> assistant -> user (with overlap) -----------
        if role == "user":
            # Look ahead for assistant
            if i + 1 < len(messages) and messages[i + 1]["speaker"] == "assistant":
                assistant_msg = messages[i + 1]
                # Look ahead for next user (overlap anchor)
                if i + 2 < len(messages) and messages[i + 2]["speaker"] == "user":
                    chunk_msgs = [current, assistant_msg, messages[i + 2]]
                    chunks.append(_make_chunk(chunk_msgs, conversation_id, chunk_index))
                    chunk_index += 1
                    i += 2  # advance to the overlap user (it starts next chunk)
                    continue
                else:
                    # Conversation ends after assistant reply: 2-msg final chunk
                    chunk_msgs = [current, assistant_msg]
                    chunks.append(_make_chunk(chunk_msgs, conversation_id, chunk_index))
                    chunk_index += 1
                    i += 2
                    continue
            else:
                # User with no following assistant: trailing unmatched user
                if i == len(messages) - 1:
                    # Last message is an unmatched user — pending tail
                    pending_tail = current
                    logger.info(
                        f"Trailing unmatched user message at position "
                        f"{current['position']} → pending_tail"
                    )
                    i += 1
                    continue
                else:
                    # Consecutive users or other break: emit as fallback 1-msg
                    # chunk and advance, so no messages are dropped
                    logger.warning(
                        f"Pattern break at position {i}: user followed by "
                        f"{messages[i+1]['speaker']} (expected assistant). "
                        "Emitting fallback single-message chunk."
                    )
                    chunks.append(_make_chunk([current], conversation_id, chunk_index))
                    chunk_index += 1
                    i += 1
                    continue

        # --- Fallback: assistant leading (e.g. after pattern break) ---------
        else:
            logger.warning(
                f"Pattern break at position {i}: assistant message without "
                "preceding user in current window. Emitting fallback chunk."
            )
            chunks.append(_make_chunk([current], conversation_id, chunk_index))
            chunk_index += 1
            i += 1
            continue

    manifest_meta = {
        "conversation_id": conversation_id,
        "title": normalized.get("title", ""),
        "source_platform": normalized.get("source_platform", "chatgpt"),
        "total_visible_messages": len(messages),
        "chunk_count": len(chunks),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "pending_tail": _pending_tail_summary(pending_tail),
        "chunked_at": _now_iso(),
    }

    logger.info(
        f"Chunking complete: {len(chunks)} chunks, "
        f"pending_tail={'yes' if pending_tail else 'no'}"
    )
    return {
        "chunks": chunks,
        "pending_tail": pending_tail,
        "manifest_meta": manifest_meta,
    }


def _make_chunk(
    messages: List[Dict[str, Any]],
    conversation_id: str,
    chunk_index: int,
) -> Dict[str, Any]:
    """Build a single chunk dict.

    Args:
        messages: Ordered list of normalized messages for this chunk.
        conversation_id: Parent conversation ID.
        chunk_index: Zero-based index of this chunk.

    Returns:
        Chunk dict.
    """
    positions = [m["position"] for m in messages]
    message_ids = [m["message_id"] for m in messages]
    chunk_id = f"{conversation_id}_c{chunk_index:03d}"

    return {
        "conversation_id": conversation_id,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "start_position": positions[0],
        "end_position": positions[-1],
        "message_ids": message_ids,
        "chunk_text": _format_chunk_text(messages),
    }


def _pending_tail_summary(msg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Summarize a pending tail message for the manifest.

    Args:
        msg: The trailing unmatched message, or None.

    Returns:
        A small dict with position, message_id, and a content preview, or None.
    """
    if msg is None:
        return None
    preview = msg["content"][:120].replace("\n", " ")
    if len(msg["content"]) > 120:
        preview += "…"
    return {
        "message_id": msg["message_id"],
        "position": msg["position"],
        "speaker": msg["speaker"],
        "preview": preview,
        "note": "Trailing unmatched user message. Will be chunked on next re-import.",
    }


def save_chunks(
    chunks: List[Dict[str, Any]],
    pending_tail: Optional[Dict[str, Any]],
    manifest_meta: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write per-chunk files and the manifest to disk.

    Args:
        chunks: List of chunk dicts.
        pending_tail: Optional trailing unmatched message.
        manifest_meta: Top-level manifest metadata.
        output_dir: Directory to write into (created if absent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        chunk_path = output_dir / f"chunk_{chunk['chunk_index']:03d}.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)

    # Write pending tail if present
    if pending_tail:
        tail_path = output_dir / "pending_tail.json"
        tail_doc = {
            **pending_tail,
            "partial": True,
            "note": "Trailing unmatched user message. Will be chunked on next re-import.",
        }
        with open(tail_path, "w", encoding="utf-8") as f:
            json.dump(tail_doc, f, indent=2, ensure_ascii=False)

    # Write manifest
    manifest_path = output_dir.parent / "chunk_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Wrote {len(chunks)} chunk files + manifest to {output_dir.parent}"
    )
