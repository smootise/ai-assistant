"""Background fragment runner for the web pipeline job flow.

Module-level function (not a closure) so daemon threads have clean state.
Each invocation opens its own SummaryStore connection — safe for threading.
"""

import logging
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from jarvis.embedder import EmbeddingClient
from jarvis.fragmenter import Fragmenter
from jarvis.memory import MemoryLayer
from jarvis.ollama import OllamaClient
from jarvis.store import SummaryStore
from jarvis.vector_store import VectorStore


logger = logging.getLogger(__name__)

_MAX_IDS_IN_RESULT = 200


def run_fragment_job(
    job_id: str,
    conversation_id: str,
    options: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Run a fragment-extracts job and update the job row when done.

    options keys: from_segment (int), to_segment (int|None), force (bool),
                  persist (bool), embed (bool).
    Called from a daemon thread spawned by the pipeline_jobs route.
    """
    store = SummaryStore(db_path=config["db_path"])
    store.mark_job_running(job_id)
    logger.info(f"Job {job_id}: starting fragment for conversation {conversation_id}")

    try:
        from_segment: int = options.get("from_segment", 0)
        to_segment: Optional[int] = options.get("to_segment")
        force: bool = options.get("force", False)
        persist: bool = options.get("persist", False)
        embed: bool = options.get("embed", False)

        if embed and not persist:
            raise ValueError("embed requires persist to be enabled")

        output_root = Path(config.get("output_root", "OUTPUTS"))
        extract_dir = output_root / conversation_id / "extracts"
        fragment_dir = output_root / conversation_id / "fragments"

        if not extract_dir.exists():
            raise FileNotFoundError(
                f"Extracts directory not found: {extract_dir}. "
                "Run extract-segments first."
            )

        if force:
            _wipe_fragments_for_force(
                store=store,
                fragment_dir=fragment_dir,
                conversation_id=conversation_id,
                config=config,
            )

        ollama_client = OllamaClient(
            model=config["local_model_name"],
            base_url=config["ollama_base_url"],
            timeout=config["ollama_timeout"],
        )
        fragmenter = Fragmenter(
            ollama_client=ollama_client,
            prompts_dir=config["prompts_dir"],
            schema=config["schema"],
            schema_version=config["schema_version"],
        )

        results, skipped_segments = fragmenter.fragment_conversation_extracts(
            conversation_id=conversation_id,
            output_root=output_root,
            from_segment=from_segment,
            to_segment=to_segment,
            force=force,
            retries=config.get("retries", 1),
        )

        fragments_persisted = 0
        fragment_ids: List[str] = []

        if persist and results:
            memory = _build_memory_layer(config)
            for _, output_data in results:
                segment_id = output_data.get("segment_id")
                frag_idx = output_data.get("fragment_index")
                if segment_id is None or frag_idx is None:
                    continue

                extract_id = f"{segment_id}_x"
                fragment_id = f"{extract_id}_f{frag_idx:03d}"

                existing = memory.store.get_fragment(fragment_id)
                if existing:
                    # Repair missing links if needed (mirrors cli.py behavior)
                    link_count = memory.store.get_link_count(fragment_id)
                    if link_count == 0 and output_data.get("statements"):
                        frag_stmts = output_data["statements"]
                        if isinstance(frag_stmts[0].get("statement_index"), int):
                            statement_ids = [
                                f"{extract_id}_st{s['statement_index']:04d}"
                                for s in frag_stmts
                            ]
                        else:
                            db_stmts = memory.store.get_statements_for_extract(extract_id)
                            text_to_id = {s["text"]: s["statement_id"] for s in db_stmts}
                            statement_ids = [
                                text_to_id[s["text"]]
                                for s in frag_stmts
                                if s["text"] in text_to_id
                            ]
                        if statement_ids:
                            memory.store.insert_fragment_links(fragment_id, statement_ids)

                    if embed and not existing.get("qdrant_point_id"):
                        memory.index_fragment_in_qdrant(
                            fragment_id=fragment_id, output_data=output_data
                        )
                        fragments_persisted += 1
                    fragment_ids.append(fragment_id)
                    continue

                fid = memory.persist_fragment_with_links(
                    output_data=output_data, extract_id=extract_id
                )
                if embed:
                    memory.index_fragment_in_qdrant(
                        fragment_id=fid, output_data=output_data
                    )
                fragment_ids.append(fid)
                fragments_persisted += 1

        result = {
            "conversation_id": conversation_id,
            "fragments_produced": len(results),
            "fragments_persisted": fragments_persisted,
            "embedded": embed and persist,
            "skipped_segments": [[sid, reason] for sid, reason in skipped_segments],
            "fragment_ids": fragment_ids[:_MAX_IDS_IN_RESULT],
            "fragment_ids_truncated": len(fragment_ids) > _MAX_IDS_IN_RESULT,
        }
        store.mark_job_succeeded(job_id, result)
        logger.info(
            f"Job {job_id}: fragment succeeded — {len(results)} fragments, "
            f"{fragments_persisted} persisted, embedded={embed and persist}"
        )

    except Exception:
        error_text = traceback.format_exc()
        store.mark_job_failed(job_id, error_text)
        logger.error(f"Job {job_id}: fragment failed\n{error_text}")


def _wipe_fragments_for_force(
    store: SummaryStore,
    fragment_dir: Path,
    conversation_id: str,
    config: Dict[str, Any],
) -> None:
    """Delete existing fragment files, DB rows, and Qdrant points for the conversation."""
    point_ids = store.delete_fragments(conversation_id)
    if point_ids:
        try:
            vs = VectorStore(host=config["qdrant_host"], port=config["qdrant_port"])
            vs.delete_points(point_ids)
        except Exception:
            pass

    if fragment_dir.exists():
        for seg_dir in fragment_dir.glob("segment_*"):
            if seg_dir.is_dir():
                shutil.rmtree(seg_dir)

    logger.info(f"--force: wiped existing fragments for {conversation_id}")


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
