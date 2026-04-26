"""Background extract runner for the web pipeline job flow.

Module-level function (not a closure) so daemon threads have clean state.
Each invocation opens its own SummaryStore connection — safe for threading.
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from jarvis.embedder import EmbeddingClient
from jarvis.extractor import SegmentExtractor
from jarvis.memory import MemoryLayer
from jarvis.ollama import OllamaClient
from jarvis.store import SummaryStore
from jarvis.vector_store import VectorStore


logger = logging.getLogger(__name__)

# Cap on extract_ids stored in the result blob to avoid unbounded JSON.
_MAX_IDS_IN_RESULT = 200


def run_extract_job(
    job_id: str,
    conversation_id: str,
    options: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Run an extract-segments job and update the job row when done.

    options keys: from_segment (int), to_segment (int|None), force (bool), persist (bool).
    Called from a daemon thread spawned by the pipeline_jobs route.
    """
    store = SummaryStore(db_path=config["db_path"])
    store.mark_job_running(job_id)
    logger.info(f"Job {job_id}: starting extract for conversation {conversation_id}")

    try:
        from_segment: int = options.get("from_segment", 0)
        to_segment: Optional[int] = options.get("to_segment")
        force: bool = options.get("force", False)
        persist: bool = options.get("persist", False)

        repo_root = Path(config.get("repo_root", "."))
        inbox_dir = repo_root / "inbox" / "ai_chat" / "chatgpt"
        segments_dir = inbox_dir / conversation_id / "segments"
        output_root = Path(config.get("output_root", "OUTPUTS"))
        extract_dir = output_root / conversation_id / "extracts"

        if not segments_dir.exists():
            raise FileNotFoundError(
                f"Segments directory not found: {segments_dir}. Run ingest first."
            )

        if force:
            _wipe_extracts_for_force(
                store=store,
                segments_dir=segments_dir,
                extract_dir=extract_dir,
                conversation_id=conversation_id,
                from_segment=from_segment,
                to_segment=to_segment,
                persist=persist,
            )

        ollama_client = OllamaClient(
            model=config["local_model_name"],
            base_url=config["ollama_base_url"],
            timeout=config["ollama_timeout"],
        )
        extractor = SegmentExtractor(
            ollama_client=ollama_client,
            prompts_dir=config["prompts_dir"],
            schema=config["schema"],
            schema_version=config["schema_version"],
        )

        results = extractor.extract_conversation_segments(
            segments_dir=segments_dir,
            conversation_id=conversation_id,
            output_root=output_root,
            from_segment=from_segment,
            to_segment=to_segment,
            force=force,
            retries=config.get("retries", 1),
        )

        segments_processed = len(results)
        extracts_persisted = 0
        extract_ids: List[str] = []

        if persist and results:
            memory = _build_memory_layer(config)
            for _, output_data in results:
                segment_id = output_data.get("segment_id")
                if not segment_id:
                    continue
                existing = memory.store.get_extract_by_segment(segment_id)
                if existing:
                    extract_ids.append(existing["extract_id"])
                    continue
                eid = memory.persist_extract_with_statements(output_data)
                extract_ids.append(eid)
                extracts_persisted += 1

        result = {
            "conversation_id": conversation_id,
            "segments_processed": segments_processed,
            "extracts_persisted": extracts_persisted,
            "extract_ids": extract_ids[:_MAX_IDS_IN_RESULT],
            "extract_ids_truncated": len(extract_ids) > _MAX_IDS_IN_RESULT,
        }
        store.mark_job_succeeded(job_id, result)
        logger.info(
            f"Job {job_id}: extract succeeded — {segments_processed} segments, "
            f"{extracts_persisted} persisted"
        )

    except Exception:
        error_text = traceback.format_exc()
        store.mark_job_failed(job_id, error_text)
        logger.error(f"Job {job_id}: extract failed\n{error_text}")


def _wipe_extracts_for_force(
    store: SummaryStore,
    segments_dir: Path,
    extract_dir: Path,
    conversation_id: str,
    from_segment: int,
    to_segment: Optional[int],
    persist: bool,
) -> None:
    """Delete existing extract files and DB rows for the forced range."""
    all_seg_indices = sorted(
        int(f.stem.split("_")[1])
        for f in segments_dir.glob("segment_*.json")
        if f.name != "pending_tail.json"
    )
    effective_to = (
        to_segment if to_segment is not None
        else (max(all_seg_indices) if all_seg_indices else from_segment)
    )
    forced_indices = list(range(from_segment, effective_to + 1))

    if extract_dir.exists():
        for idx in forced_indices:
            for suffix in (".json", ".md"):
                p = extract_dir / f"extract_{idx:03d}{suffix}"
                if p.exists():
                    p.unlink()

    if persist:
        store.delete_extracts(conversation_id, segment_indices=forced_indices)

    logger.info(
        f"--force: wiped extracts {from_segment}–{effective_to} for {conversation_id}"
    )


def _build_memory_layer(config: Dict[str, Any]) -> MemoryLayer:
    store = SummaryStore(db_path=config["db_path"])
    vector_store = VectorStore(
        host=config["qdrant_host"],
        port=config["qdrant_port"],
    )
    embedder = EmbeddingClient(
        model=config["embedding_model"],
        base_url=config["ollama_base_url"],
    )
    return MemoryLayer(store=store, vector_store=vector_store, embedder=embedder)
