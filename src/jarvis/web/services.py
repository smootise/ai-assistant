"""Service layer for the web UI.

Each function takes a SummaryStore and returns a plain dict for template rendering.
Missing linked records are normalized to None / [] — pages never raise from absent data.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jarvis.store import SummaryStore
from jarvis.cli import build_memory_layer, run_retrieval, generate_answer
from jarvis.ollama import OllamaClient


logger = logging.getLogger(__name__)


def get_dashboard_data(store: SummaryStore) -> Dict[str, Any]:
    counts = store.count_all()
    recent = store.recent_records(10)
    return {"counts": counts, "recent": recent}


def get_sources_list(
    store: SummaryStore, limit: int = 100, offset: int = 0
) -> List[Dict[str, Any]]:
    return store.list_source_files(limit=limit, offset=offset)


def get_source_detail(store: SummaryStore, source_id: str) -> Optional[Dict[str, Any]]:
    source = store.get_source_file(source_id)
    if source is None:
        return None
    conversations = store.list_conversations_for_source(source_id)
    return {"source": source, "conversations": conversations}


def get_conversations_list(
    store: SummaryStore, limit: int = 100, offset: int = 0
) -> List[Dict[str, Any]]:
    return store.list_conversations(limit=limit, offset=offset)


def get_conversation_detail(
    store: SummaryStore, conversation_id: str
) -> Optional[Dict[str, Any]]:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        return None
    segments = store.list_segments_for_conversation(conversation_id)
    has_extracts = store.count_extracts_for_conversation(conversation_id) > 0
    return {"conversation": conversation, "segments": segments, "has_extracts": has_extracts}


def get_segment_detail(store: SummaryStore, segment_id: str) -> Optional[Dict[str, Any]]:
    segment = store.get_segment(segment_id)
    if segment is None:
        return None
    conversation = store.get_conversation(segment["conversation_id"])
    extract = store.get_extract_by_segment(segment_id)
    return {"segment": segment, "conversation": conversation, "extract": extract}


def get_extract_detail(store: SummaryStore, extract_id: str) -> Optional[Dict[str, Any]]:
    extract = store.get_extract(extract_id)
    if extract is None:
        return None
    statements = store.get_statements_for_extract(extract_id)
    fragments = store.list_fragments_for_extract(extract_id)
    segment = store.get_segment(extract["segment_id"])
    conversation = (
        store.get_conversation(extract["parent_conversation_id"])
        if extract.get("parent_conversation_id") else None
    )
    return {
        "extract": extract,
        "statements": statements,
        "fragments": fragments,
        "segment": segment,
        "conversation": conversation,
    }


def get_fragment_detail(store: SummaryStore, fragment_id: str) -> Optional[Dict[str, Any]]:
    results = store.get_fragments_with_statements([fragment_id])
    if not results:
        return None
    fragment = results[0]
    extract = store.get_extract(fragment["extract_id"])
    return {"fragment": fragment, "extract": extract}


def run_search(
    config: Dict[str, Any],
    query: str,
    top_k: int = 10,
    min_results: int = 3,
    min_score: float = 0.50,
) -> Dict[str, Any]:
    """Run vector retrieval and return ranked fragment results.

    Returns {"results": [...], "error": None|str}.
    Each result has: rank, fragment_id, score, title, snippet, segment_id,
    extract_id, parent_conversation_id, conversation_date.
    """
    try:
        memory = build_memory_layer(config)
        rows, scores = run_retrieval(memory, query, top_k, min_results, min_score)
    except (ConnectionError, RuntimeError) as exc:
        logger.error(f"Search failed: {exc}")
        return {"results": [], "error": str(exc)}
    except Exception as exc:
        logger.exception(f"Unexpected search error: {exc}")
        return {"results": [], "error": f"Unexpected error: {exc}"}

    results = []
    for rank, row in enumerate(rows, start=1):
        fid = row["fragment_id"]
        title = row.get("title") or ""
        stmts = row.get("statements", [])
        body = " / ".join(f"{s['speaker']}: {s['text']}" for s in stmts[:2])
        raw_snippet = f"[{title}] {body}" if title else body
        snippet = raw_snippet[:160].replace("\n", " ")
        if len(raw_snippet) > 160:
            snippet += "..."
        results.append({
            "rank": rank,
            "fragment_id": fid,
            "score": scores.get(fid, 0.0),
            "title": title,
            "snippet": snippet,
            "segment_id": row.get("segment_id"),
            "extract_id": row.get("extract_id"),
            "parent_conversation_id": row.get("parent_conversation_id"),
            "conversation_date": row.get("conversation_date"),
        })
    return {"results": results, "error": None}


def run_answer(
    config: Dict[str, Any],
    query: str,
    top_k: int = 10,
    min_results: int = 3,
    min_score: float = 0.50,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Run retrieval then generate an LLM answer with citations.

    Returns {"query", "answer", "citations", "is_degraded", "warning", "error"}.
    citations: [{rank, fragment_id, title, segment_id, conversation_date}].
    """
    base: Dict[str, Any] = {
        "query": query,
        "answer": "",
        "citations": [],
        "is_degraded": False,
        "warning": "",
        "error": None,
    }
    try:
        memory = build_memory_layer(config)
        rows, _ = run_retrieval(memory, query, top_k, min_results, min_score)
    except (ConnectionError, RuntimeError) as exc:
        logger.error(f"Retrieval failed: {exc}")
        return {**base, "error": str(exc)}
    except Exception as exc:
        logger.exception(f"Unexpected retrieval error: {exc}")
        return {**base, "error": f"Unexpected error: {exc}"}

    if not rows:
        return base

    citations = [
        {
            "rank": i,
            "fragment_id": row["fragment_id"],
            "title": row.get("title") or "",
            "segment_id": row.get("segment_id"),
            "conversation_date": row.get("conversation_date"),
        }
        for i, row in enumerate(rows, start=1)
    ]

    try:
        ollama = OllamaClient(
            base_url=config["ollama_base_url"],
            model=config["local_model_name"],
            timeout=config["ollama_timeout"],
        )
        answer, is_degraded, warning = generate_answer(
            memory, ollama, query, rows,
            config["prompts_dir"], config.get("user_name", ""), temperature
        )
    except (ConnectionError, RuntimeError) as exc:
        logger.error(f"Answer generation failed: {exc}")
        return {**base, "citations": citations, "error": str(exc)}
    except Exception as exc:
        logger.exception(f"Unexpected answer error: {exc}")
        return {**base, "citations": citations, "error": f"Unexpected error: {exc}"}

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "is_degraded": is_degraded,
        "warning": warning,
        "error": None,
    }


def save_upload(
    file_storage: Any,
    safe_name: str,
    raw_dir: Path,
) -> Tuple[Optional[Path], Dict[str, Any], Optional[str]]:
    """Save an uploaded file to raw_dir with a UTC timestamp prefix.

    Returns (saved_path, input_metadata, error_string). On error, saved_path is
    None and error_string is set. On success, error_string is None.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{ts}_{safe_name}"
    dest = raw_dir / filename

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        file_storage.save(str(dest))
    except OSError as exc:
        logger.error(f"Failed to save upload to {dest}: {exc}")
        return None, {}, f"Could not save file: {exc}"

    input_metadata = {
        "original_filename": safe_name,
        "stored_filename": filename,
        "storage_path": str(dest),
        "source_type": "chatgpt",
    }
    return dest, input_metadata, None
