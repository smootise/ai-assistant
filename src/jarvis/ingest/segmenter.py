"""Segmenter for normalized AI chat conversations.

Implements the source-specific `ai_chat` segmentation strategy:
  user -> assistant -> user (with overlap on the trailing user message)

Segment i:   [user_n, assistant_n, user_{n+1}]
Segment i+1: [user_{n+1}, assistant_{n+1}, user_{n+2}]

Special cases:
- If the conversation ends after an assistant reply, emit a final
  2-message segment (user, assistant).
- A trailing unmatched user message is NOT emitted as a normal segment.
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


def _format_segment_text(messages: List[Dict[str, Any]]) -> str:
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


def segment_conversation(
    normalized: Dict[str, Any],
) -> Dict[str, Any]:
    """Segment a normalized conversation into summarization-ready segments.

    Args:
        normalized: Normalized conversation dict (from normalizer.py).

    Returns:
        Dict with keys:
            segments: list of segment dicts
            pending_tail: optional single message dict (trailing unmatched user)
            manifest_meta: top-level metadata for the manifest
    """
    messages = normalized.get("messages", [])
    conversation_id = normalized.get("conversation_id", "")
    segments: List[Dict[str, Any]] = []
    pending_tail: Optional[Dict[str, Any]] = None

    i = 0
    segment_index = 0

    while i < len(messages):
        current = messages[i]
        role = current["speaker"]

        # --- Ideal path: user -> assistant(s) -> user (with overlap) ---------
        if role == "user":
            # Look ahead for at least one assistant message
            if i + 1 < len(messages) and messages[i + 1]["speaker"] == "assistant":
                # Collect all consecutive assistant messages (thinking-mode preambles
                # produce a short "I'm looking into this…" message immediately followed
                # by the full answer — keep them together in one segment).
                j = i + 1
                while j < len(messages) and messages[j]["speaker"] == "assistant":
                    j += 1
                # messages[i+1 .. j-1] are all assistant turns
                assistant_msgs = messages[i + 1:j]

                # Look ahead for the next user message (overlap anchor)
                if j < len(messages) and messages[j]["speaker"] == "user":
                    segment_msgs = [current] + assistant_msgs + [messages[j]]
                    segments.append(_make_segment(segment_msgs, conversation_id, segment_index))
                    segment_index += 1
                    i = j  # advance to the overlap user (it starts next segment)
                    continue
                else:
                    # Conversation ends after the assistant turn(s): final segment
                    segment_msgs = [current] + assistant_msgs
                    segments.append(_make_segment(segment_msgs, conversation_id, segment_index))
                    segment_index += 1
                    i = j
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
                    # segment and advance, so no messages are dropped
                    logger.warning(
                        f"Pattern break at position {i}: user followed by "
                        f"{messages[i + 1]['speaker']} (expected assistant). "
                        "Emitting fallback single-message segment."
                    )
                    segments.append(_make_segment([current], conversation_id, segment_index))
                    segment_index += 1
                    i += 1
                    continue

        # --- Fallback: assistant leading (e.g. start of export or pattern break) ---
        else:
            # Collect all consecutive assistant messages at this position
            j = i
            while j < len(messages) and messages[j]["speaker"] == "assistant":
                j += 1
            orphan_msgs = messages[i:j]
            logger.warning(
                f"Orphaned assistant message(s) at position {i} "
                f"(no preceding user). Emitting fallback segment of {len(orphan_msgs)} msg(s)."
            )
            segments.append(_make_segment(orphan_msgs, conversation_id, segment_index))
            segment_index += 1
            i = j
            continue

    manifest_meta = {
        "conversation_id": conversation_id,
        "title": normalized.get("title", ""),
        "source_platform": normalized.get("source_platform", "chatgpt"),
        "total_visible_messages": len(messages),
        "segment_count": len(segments),
        "segment_ids": [s["segment_id"] for s in segments],
        "pending_tail": _pending_tail_summary(pending_tail),
        "segmented_at": _now_iso(),
    }

    logger.info(
        f"Segmentation complete: {len(segments)} segments, "
        f"pending_tail={'yes' if pending_tail else 'no'}"
    )
    return {
        "segments": segments,
        "pending_tail": pending_tail,
        "manifest_meta": manifest_meta,
    }


def _make_segment(
    messages: List[Dict[str, Any]],
    conversation_id: str,
    segment_index: int,
) -> Dict[str, Any]:
    """Build a single segment dict.

    Args:
        messages: Ordered list of normalized messages for this segment.
        conversation_id: Parent conversation ID.
        segment_index: Zero-based index of this segment.

    Returns:
        Segment dict.
    """
    positions = [m["position"] for m in messages]
    message_ids = [m["message_id"] for m in messages]
    segment_id = f"{conversation_id}_s{segment_index:03d}"

    # Use the timestamp of the first message as the segment's conversation date.
    # This is when the exchange actually happened, not when the pipeline ran.
    conversation_date = messages[0].get("created_at")

    return {
        "conversation_id": conversation_id,
        "segment_id": segment_id,
        "segment_index": segment_index,
        "start_position": positions[0],
        "end_position": positions[-1],
        "message_ids": message_ids,
        "conversation_date": conversation_date,
        "segment_text": _format_segment_text(messages),
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
        "note": "Trailing unmatched user message. Will be segmented on next re-import.",
    }


def save_segments(
    segments: List[Dict[str, Any]],
    pending_tail: Optional[Dict[str, Any]],
    manifest_meta: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write per-segment files and the manifest to disk.

    Args:
        segments: List of segment dicts.
        pending_tail: Optional trailing unmatched message.
        manifest_meta: Top-level manifest metadata.
        output_dir: Directory to write into (created if absent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        segment_path = output_dir / f"segment_{segment['segment_index']:03d}.json"
        with open(segment_path, "w", encoding="utf-8") as f:
            json.dump(segment, f, indent=2, ensure_ascii=False)

    # Write pending tail if present
    if pending_tail:
        tail_path = output_dir / "pending_tail.json"
        tail_doc = {
            **pending_tail,
            "partial": True,
            "note": "Trailing unmatched user message. Will be segmented on next re-import.",
        }
        with open(tail_path, "w", encoding="utf-8") as f:
            json.dump(tail_doc, f, indent=2, ensure_ascii=False)

    # Write manifest
    manifest_path = output_dir.parent / "segment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Wrote {len(segments)} segment files + manifest to {output_dir.parent}"
    )
