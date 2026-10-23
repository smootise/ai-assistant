"""Exercise real SQLite persistence with mocked inference and unavailable Qdrant."""

import argparse
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from jarvis import cli
from jarvis.clients import build_memory_layer, create_summary_store
from jarvis.config import load_config
from jarvis.store import SummaryStore
from jarvis.web.extract_runner import run_extract_job
from jarvis.web.fragment_runner import run_fragment_job


@pytest.fixture
def persistence_config(profile_repo: Path) -> dict[str, Any]:
    config = load_config()
    store = create_summary_store(config)
    store.insert_conversation({"conversation_id": "synthetic"})
    store.insert_segment(
        {
            "segment_id": "synthetic_s000",
            "conversation_id": "synthetic",
            "segment_index": 0,
            "start_position": 0,
            "end_position": 1,
            "message_ids": ["m0"],
            "segment_text": "A synthetic statement.",
        }
    )
    (Path(config["chatgpt_inbox_dir"]) / "synthetic" / "segments").mkdir(parents=True)
    for name in ("extracts", "segment_summaries"):
        (Path(config["output_root"]) / "synthetic" / name).mkdir(parents=True)
    return config


def output_data() -> dict[str, Any]:
    return {
        "segment_id": "synthetic_s000",
        "segment_index": 0,
        "parent_conversation_id": "synthetic",
        "provider": "local",
        "model": "synthetic-model",
        "status": "ok",
        "summary": "Synthetic summary",
        "topic_index": 0,
        "topic_segment_range": "s000-s000",
        "fragment_index": 0,
        "title": "Synthetic fragment",
        "text": "A synthetic statement.",
        "statements": [{"statement_index": 0, "speaker": "user", "text": "A synthetic statement."}],
    }


def seed_extract(store: SummaryStore) -> None:
    data = output_data()
    extract_id = store.insert_extract(data)
    store.insert_statements(extract_id, data["statements"])


def assert_persisted(store: SummaryStore, stage: str) -> None:
    if stage == "summarize_segments":
        assert store.get_segment_summary("synthetic_s000")["summary"] == "Synthetic summary"
    elif stage == "extract_segments":
        extract = store.get_extract_by_segment("synthetic_s000")
        statements = store.get_statements_for_extract(extract["extract_id"])
        assert len(statements) == 1
        assert statements[0]["text"] == "A synthetic statement."
    elif stage == "fragment_extracts":
        fragment_id = "synthetic_s000_x_f000"
        fragment = store.get_fragment(fragment_id)
        assert fragment["retrieval_text"] == "A synthetic statement."
        assert fragment["qdrant_point_id"] is None
        assert fragment["embedding_model"] is None
        assert store.get_link_count(fragment_id) == 1
    else:
        with closing(store._connect()) as conn:
            assert conn.execute("SELECT topic_id, summary FROM topic_summaries").fetchall() == [
                ("synthetic_t000", "Synthetic summary")
            ]
            assert conn.execute("SELECT topic_id, segment_id FROM topic_segments").fetchall() == [
                ("synthetic_t000", "synthetic_s000")
            ]


@pytest.mark.parametrize(
    "stage,driver,method",
    [
        ("summarize_segments", "SegmentSummarizer", "summarize_conversation_segments"),
        ("detect_topics", "TopicDetector", "detect_and_summarize"),
        ("extract_segments", "SegmentExtractor", "extract_conversation_segments"),
        ("fragment_extracts", "Fragmenter", "fragment_conversation_extracts"),
    ],
)
def test_cli_sqlite_persistence_succeeds_without_qdrant(
    persistence_config: dict[str, Any], stage: str, driver: str, method: str
) -> None:
    config = persistence_config
    store = create_summary_store(config)
    if stage == "fragment_extracts":
        seed_extract(store)
    args = argparse.Namespace(
        source="chatgpt",
        conversation_id="synthetic",
        inbox_dir=config["chatgpt_inbox_dir"],
        persist=True,
        embed=False,
        force=False,
        from_segment=0,
        to_segment=None,
        context_window=2,
        threshold=0.5,
        dry_run=False,
    )
    results = [(Path(config["output_root"]), output_data())]
    with (
        patch(
            "jarvis.vector_store.QdrantClient", side_effect=RuntimeError("Qdrant unavailable")
        ) as qdrant,
        patch("jarvis.clients.EmbeddingClient") as embeddings,
        patch("jarvis.clients.OllamaClient"),
        patch(f"jarvis.cli.{driver}") as pipeline,
    ):
        getattr(pipeline.return_value, method).return_value = (
            (results, []) if stage == "fragment_extracts" else results
        )
        for _ in range(2):
            assert getattr(cli, f"cmd_{stage}")(args, config) == 0
            assert_persisted(store, stage)
        qdrant.assert_not_called()
        # Topic detection embeds summary text, independently of SQLite persistence.
        assert embeddings.call_count == (2 if stage == "detect_topics" else 0)


@pytest.mark.parametrize(
    "stage,runner,driver,method",
    [
        ("extract_segments", run_extract_job, "SegmentExtractor", "extract_conversation_segments"),
        ("fragment_extracts", run_fragment_job, "Fragmenter", "fragment_conversation_extracts"),
    ],
)
def test_web_sqlite_jobs_succeed_without_qdrant(
    persistence_config: dict[str, Any], stage: str, runner: Any, driver: str, method: str
) -> None:
    config = persistence_config
    store = create_summary_store(config)
    if stage == "fragment_extracts":
        seed_extract(store)
    module = "extract_runner" if stage == "extract_segments" else "fragment_runner"
    results = [(Path(config["output_root"]), output_data())]
    with (
        patch(
            "jarvis.vector_store.QdrantClient", side_effect=RuntimeError("Qdrant unavailable")
        ) as qdrant,
        patch("jarvis.clients.EmbeddingClient") as embeddings,
        patch("jarvis.clients.OllamaClient"),
        patch(f"jarvis.web.{module}.{driver}") as pipeline,
    ):
        getattr(pipeline.return_value, method).return_value = (
            (results, []) if stage == "fragment_extracts" else results
        )
        for _ in range(2):
            job_id = store.create_job(stage, {"conversation_id": "synthetic"})
            runner(job_id, "synthetic", {"persist": True, "embed": False, "force": False}, config)
            assert store.get_job(job_id)["status"] == "succeeded"
            assert_persisted(store, stage)
        qdrant.assert_not_called()
        embeddings.assert_not_called()


@pytest.mark.parametrize("unavailable", [False, True])
def test_vector_clients_initialize_only_for_indexing_and_keep_resolved_configuration(
    persistence_config: dict[str, Any], unavailable: bool
) -> None:
    config = dict(persistence_config)
    config["embedding_ollama_base_url"] = "http://synthetic-embedding:11434"
    with (
        patch("jarvis.clients.VectorStore") as vectors,
        patch("jarvis.clients.EmbeddingClient") as embeddings,
    ):
        if unavailable:
            vectors.side_effect = RuntimeError("Qdrant unavailable")
        else:
            vectors.return_value.upsert.return_value = "synthetic-point"
        embeddings.return_value.embed.return_value = [1.0, 0.0]
        embeddings.return_value.model = config["embedding_model"]
        memory = build_memory_layer(config)
        data = output_data()
        extract_id = memory.persist_extract_with_statements(data)
        fragment_id = memory.persist_fragment_with_links(data, extract_id)
        vectors.assert_not_called()
        embeddings.assert_not_called()
        config["qdrant_collection"] = "later-profile-collection"
        config["embedding_ollama_base_url"] = "http://later-embedding:11434"

        if unavailable:
            with pytest.raises(RuntimeError, match="Qdrant unavailable"):
                memory.index_fragment_in_qdrant(fragment_id, data)
            assert memory.store.get_fragment(fragment_id)["qdrant_point_id"] is not None
            assert memory.store.get_fragment(fragment_id)["embedded_at"] is None
        else:
            for _ in range(2):
                assert memory.index_fragment_in_qdrant(fragment_id, data) == "synthetic-point"
            assert memory.store.get_fragment(fragment_id)["qdrant_point_id"] == "synthetic-point"
        vectors.assert_called_once_with(
            host="localhost",
            port=6333,
            collection_name="jarvis_fragments",
            api_key=None,
        )
        embeddings.assert_called_once_with(
            model=config["embedding_model"],
            base_url="http://synthetic-embedding:11434",
        )
