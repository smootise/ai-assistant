"""Helper that seeds a SummaryStore with a minimal graph for web tests.

Graph: 1 source → 1 conversation → 2 segments → 1 extract (on seg 0)
       with 3 statements → 2 fragments with links.
"""

from jarvis.store import SummaryStore


def seed(store: SummaryStore) -> dict:
    """Seed the store and return the IDs created."""
    source_id = "src_test_001"
    store.insert_source_file({
        "source_file_id": source_id,
        "source_kind": "chatgpt",
        "original_filename": "test_export.json",
        "storage_path": "inbox/ai_chat/chatgpt/conv_test_001/normalized.json",
        "sha256": "aabbccdd" * 8,
        "size_bytes": 1024,
        "ingested_at": "2024-01-01T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
    })

    conv_id = "conv_test_001"
    store.insert_conversation({
        "conversation_id": conv_id,
        "raw_source_file_id": source_id,
        "normalized_source_file_id": source_id,
        "title": "Test Conversation",
        "conversation_date": "2024-01-01",
        "source_platform": "chatgpt",
        "message_count": 4,
        "imported_at": "2024-01-01T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
    })

    seg0_id = store.insert_segment({
        "segment_id": f"{conv_id}_seg000",
        "conversation_id": conv_id,
        "segment_index": 0,
        "start_position": 0,
        "end_position": 2,
        "message_ids": ["m0", "m1"],
        "conversation_date": "2024-01-01",
        "segment_text": "User: Hello\nAssistant: Hi there!",
        "created_at": "2024-01-01T00:00:01Z",
    })

    seg1_id = store.insert_segment({
        "segment_id": f"{conv_id}_seg001",
        "conversation_id": conv_id,
        "segment_index": 1,
        "start_position": 2,
        "end_position": 4,
        "message_ids": ["m2", "m3"],
        "conversation_date": "2024-01-01",
        "segment_text": "User: Another question\nAssistant: Another answer",
        "created_at": "2024-01-01T00:00:02Z",
    })

    extract_id = store.insert_extract({
        "segment_id": seg0_id,
        "segment_index": 0,
        "parent_conversation_id": conv_id,
        "provider": "local",
        "model": "test-model",
        "status": "ok",
        "created_at": "2024-01-01T00:00:03Z",
    })

    statements = [
        {"statement_index": 0, "speaker": "User", "text": "Hello"},
        {"statement_index": 1, "speaker": "Assistant", "text": "Hi there!"},
        {"statement_index": 2, "speaker": "User", "text": "How are you?"},
    ]
    store.insert_statements(extract_id, statements)

    # derive statement IDs the same way store does
    st_ids = [f"{extract_id}_st{s['statement_index']:04d}" for s in statements]

    frag0_id = store.insert_fragment({
        "extract_id": extract_id,
        "fragment_index": 0,
        "title": "Greeting exchange",
        "retrieval_text": "User: Hello\nAssistant: Hi there!",
        "status": "ok",
        "created_at": "2024-01-01T00:00:04Z",
    })
    store.insert_fragment_links(frag0_id, [st_ids[0], st_ids[1]])

    frag1_id = store.insert_fragment({
        "extract_id": extract_id,
        "fragment_index": 1,
        "title": "Follow-up",
        "retrieval_text": "User: How are you?",
        "status": "ok",
        "created_at": "2024-01-01T00:00:05Z",
    })
    store.insert_fragment_links(frag1_id, [st_ids[2]])

    return {
        "source_id": source_id,
        "conv_id": conv_id,
        "seg0_id": seg0_id,
        "seg1_id": seg1_id,
        "extract_id": extract_id,
        "frag0_id": frag0_id,
        "frag1_id": frag1_id,
        "statement_ids": st_ids,
    }
