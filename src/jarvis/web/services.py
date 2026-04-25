"""Service layer for the web UI.

Each function takes a SummaryStore and returns a plain dict for template rendering.
Missing linked records are normalized to None / [] — pages never raise from absent data.
"""

from typing import Any, Dict, List, Optional

from jarvis.store import SummaryStore


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
    return {"conversation": conversation, "segments": segments}


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
