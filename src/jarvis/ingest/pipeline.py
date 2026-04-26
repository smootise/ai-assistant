"""Reusable ingest orchestration for the ChatGPT export pipeline.

Extracted from cmd_ingest in cli.py so the web upload flow can call it
without subprocess. The CLI thin-shell still exists; this module owns
all business logic.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from jarvis.ingest.chatgpt_parser import load_raw_export, parse_export
from jarvis.ingest.normalizer import (
    build_normalized,
    load_normalized,
    merge_normalized,
    save_normalized,
)
from jarvis.ingest.segmenter import save_segments, segment_conversation


logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def source_file_data(path: Path, source_kind: str) -> Dict[str, Any]:
    sha = sha256_file(path)
    return {
        "source_file_id": sha,
        "source_kind": source_kind,
        "original_filename": path.name,
        "storage_path": str(path),
        "sha256": sha,
        "size_bytes": path.stat().st_size,
    }


def ingest_chatgpt(
    raw_path: Path,
    output_dir: Path,
    persist: bool,
    config: Dict[str, Any],
    memory: Optional[Any] = None,
    store: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the full ChatGPT ingest pipeline for one export file.

    Returns a dict: {conversation_id, source_file_id, segment_count, normalized_path}.
    Raises ValueError on parse errors, FileNotFoundError if raw_path is missing.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw export file not found: {raw_path}")

    logger.info(f"Loading raw export: {raw_path}")
    raw = load_raw_export(str(raw_path))
    messages = parse_export(raw)
    logger.info(f"Parsed {len(messages)} visible messages")

    conversation_id = raw.get("conversation_id", raw_path.stem)
    conv_dir = output_dir / conversation_id
    norm_path = conv_dir / "normalized.json"

    existing = load_normalized(norm_path)
    if existing:
        logger.info("Existing normalized file found — merging")
        normalized = merge_normalized(existing, messages)
    else:
        logger.info("No existing normalized file — building fresh")
        normalized = build_normalized(raw, messages, str(raw_path))

    save_normalized(normalized, norm_path)

    segments_dir = conv_dir / "segments"
    result = segment_conversation(normalized)
    save_segments(
        segments=result["segments"],
        pending_tail=result["pending_tail"],
        manifest_meta=result["manifest_meta"],
        output_dir=segments_dir,
    )

    source_file_id = sha256_file(raw_path)

    if persist:
        # Ingest only needs SQLite — Qdrant is never touched here.
        # If a MemoryLayer is provided (e.g. for testing), use it as before.
        # If a bare SummaryStore is provided, call it directly.
        # Otherwise build a SummaryStore — no VectorStore or Ollama needed.
        logger.info("Persisting source file metadata, conversation, and segments...")

        raw_file_data = source_file_data(raw_path, "chatgpt_raw_export")
        norm_file_data = source_file_data(norm_path, "chatgpt_normalized")
        first_seg = result["segments"][0] if result["segments"] else {}
        conv_date = first_seg.get("conversation_date")
        conv_data = {
            "conversation_id": conversation_id,
            "raw_source_file_id": raw_file_data["source_file_id"],
            "normalized_source_file_id": norm_file_data["source_file_id"],
            "title": normalized.get("title"),
            "conversation_date": conv_date,
            "source_platform": normalized.get("source_platform", "chatgpt"),
            "message_count": normalized.get("message_count"),
            "imported_at": normalized.get("imported_at"),
        }

        if memory is not None:
            memory.persist_source_file(raw_file_data)
            memory.persist_source_file(norm_file_data)
            memory.persist_conversation(conv_data)
            for seg in result["segments"]:
                memory.persist_segment(seg)
        else:
            from jarvis.store import SummaryStore
            _store = store if store is not None else SummaryStore(db_path=config["db_path"])
            _store.insert_source_file(raw_file_data)
            _store.insert_source_file(norm_file_data)
            _store.insert_conversation(conv_data)
            for seg in result["segments"]:
                _store.insert_segment(seg)

        logger.info(f"Persisted {len(result['segments'])} segment(s) for {conversation_id}")

    return {
        "conversation_id": conversation_id,
        "source_file_id": source_file_id,
        "segment_count": len(result["segments"]),
        "normalized_path": str(norm_path),
    }
