"""CLI entry point for JARVIS."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from jarvis.chunk_summarizer import ChunkSummarizer
from jarvis.config import load_config
from jarvis.segment_detector import SegmentDetector
from jarvis.embedder import EmbeddingClient
from jarvis.ingest.chatgpt_parser import load_raw_export, parse_export
from jarvis.ingest.chunker import chunk_conversation, save_chunks
from jarvis.ingest.normalizer import (
    build_normalized,
    load_normalized,
    merge_normalized,
    save_normalized,
)
from jarvis.memory import MemoryLayer
from jarvis.ollama import OllamaClient
from jarvis.output_writer import OutputWriter
from jarvis.store import SummaryStore
from jarvis.summarizer import ConversationSummarizer
from jarvis.vector_store import VectorStore


def setup_logging(log_level: str) -> None:
    """Configure logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_memory_layer(config: dict) -> MemoryLayer:
    """Instantiate the full memory stack from config.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        Configured MemoryLayer.
    """
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


def cmd_summarize(args: argparse.Namespace, config: dict) -> int:
    """Execute the summarize command.

    Output goes to OUTPUTS/<source_basename>/ (stable path, no timestamp).
    --force wipes existing output files and any SQLite/Qdrant records before re-running.

    Args:
        args: Parsed command-line arguments.
        config: Loaded configuration dictionary.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger = logging.getLogger(__name__)

    input_file = Path(args.file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    if input_file.suffix.lower() != ".json":
        logger.error(f"Unsupported file type: {input_file.suffix} (expected .json)")
        return 1

    source_basename = input_file.stem
    output_root = Path(config["output_root"])
    output_dir = output_root / source_basename
    json_path = output_dir / f"{source_basename}.json"
    md_path = output_dir / f"{source_basename}.md"

    try:
        # --force: wipe existing outputs and DB/Qdrant records
        if args.force:
            store = SummaryStore(db_path=config["db_path"])
            qdrant_point_id = store.delete_by_source_file(input_file.name)
            if qdrant_point_id:
                try:
                    vs = VectorStore(
                        host=config["qdrant_host"], port=config["qdrant_port"]
                    )
                    vs.delete_points([qdrant_point_id])
                except Exception:
                    pass
            for p in (json_path, md_path):
                if p.exists():
                    p.unlink()
            logger.info(f"--force: wiped existing outputs for {source_basename}")

        # Wipe stale DB/Qdrant even without --force (no-op if nothing exists)
        elif json_path.exists():
            store = SummaryStore(db_path=config["db_path"])
            qdrant_point_id = store.delete_by_source_file(input_file.name)
            if qdrant_point_id:
                try:
                    vs = VectorStore(
                        host=config["qdrant_host"], port=config["qdrant_port"]
                    )
                    vs.delete_points([qdrant_point_id])
                except Exception:
                    pass

        ollama_client = OllamaClient(
            model=config["local_model_name"],
            base_url=config["ollama_base_url"],
        )
        output_writer = OutputWriter(output_root=str(output_root))
        summarizer = ConversationSummarizer(
            ollama_client=ollama_client,
            output_writer=output_writer,
            prompts_dir=config["prompts_dir"],
            schema=config["schema"],
            schema_version=config["schema_version"],
        )

        output_dir, output_data = summarizer.summarize_file(
            str(input_file), subfolder=source_basename
        )
        logger.info(f"Outputs saved to: {output_dir}")

        if args.persist:
            logger.info("Persisting to SQLite + Qdrant...")
            memory = _build_memory_layer(config)
            summary_id = memory.persist(output_data=output_data, output_dir=output_dir)
            logger.info(f"Persisted as summary_id={summary_id}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


def cmd_retrieve(args: argparse.Namespace, config: dict) -> int:
    """Execute the retrieve command.

    Embeds the query, searches Qdrant for the top-k nearest summaries,
    fetches the matching records from SQLite, and prints ranked results
    to stdout.

    Args:
        args: Parsed command-line arguments.
        config: Loaded configuration dictionary.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger = logging.getLogger(__name__)

    try:
        memory = _build_memory_layer(config)

        logger.info(f"Embedding query with model={config['embedding_model']}...")
        query_vector = memory.embedder.embed(args.query)

        hits = memory.vector_store.search(query_vector=query_vector, top_k=args.top_k)

        if not hits:
            print("No results found.")
            return 0

        summary_ids = [sid for sid, _, _ in hits]
        scores = {sid: score for sid, score, _ in hits}
        rows = memory.store.get_by_ids(summary_ids)

        print(f"\nTop {len(rows)} result(s) for: \"{args.query}\"\n")
        print("─" * 72)

        for rank, row in enumerate(rows, start=1):
            sid = row["id"]
            score = scores.get(sid, 0.0)
            summary_preview = (row["summary"] or "")[:120].replace("\n", " ")
            if len(row["summary"] or "") > 120:
                summary_preview += "…"

            print(f"#{rank}  score={score:.4f}  id={sid}")
            print(f"    source : {row['source_file']}")
            print(f"    created: {row['created_at']}")
            print(f"    preview: {summary_preview}")
            print()

        return 0

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


def cmd_ingest(args: argparse.Namespace, config: dict) -> int:
    """Execute the ingest command.

    Parses a raw conversation export, normalizes it (merging on re-import),
    chunks it, and writes outputs to the configured inbox directory.

    Args:
        args: Parsed command-line arguments.
        config: Loaded configuration dictionary.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger = logging.getLogger(__name__)

    if not hasattr(args, "source") or args.source != "chatgpt":
        logger.error("Only 'chatgpt' source is supported currently.")
        return 1

    raw_path = Path(args.file)
    if not raw_path.exists():
        logger.error(f"Raw export file not found: {raw_path}")
        return 1

    try:
        # Parse
        logger.info(f"Loading raw export: {raw_path}")
        raw = load_raw_export(str(raw_path))
        messages = parse_export(raw)
        logger.info(f"Parsed {len(messages)} visible messages")

        # Normalize / merge
        conversation_id = raw.get("conversation_id", raw_path.stem)
        base_dir = Path(args.output_dir)
        conv_dir = base_dir / conversation_id
        norm_path = conv_dir / "normalized.json"

        existing = load_normalized(norm_path)
        if existing:
            logger.info("Existing normalized file found — merging")
            normalized = merge_normalized(existing, messages)
        else:
            logger.info("No existing normalized file — building fresh")
            normalized = build_normalized(raw, messages, str(raw_path))

        save_normalized(normalized, norm_path)

        # Chunk
        chunks_dir = conv_dir / "chunks"
        result = chunk_conversation(normalized)
        save_chunks(
            chunks=result["chunks"],
            pending_tail=result["pending_tail"],
            manifest_meta=result["manifest_meta"],
            output_dir=chunks_dir,
        )

        print(f"\nIngest complete for: {normalized.get('title', conversation_id)}")
        print(f"  Visible messages : {normalized['message_count']}")
        print(f"  Chunks written   : {len(result['chunks'])}")
        if result["pending_tail"]:
            print("  Pending tail     : 1 unmatched trailing user message")
        print(f"  Normalized file  : {norm_path}")
        print(f"  Chunks directory : {chunks_dir}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Parse error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during ingest: {e}")
        return 1


def cmd_summarize_chunks(args: argparse.Namespace, config: dict) -> int:
    """Execute the summarize-chunks command.

    Default behavior: skip LLM for chunks whose .json already exists, then
    persist all results (new + existing) at the end.

    --force (scoped to --from-chunk/--to-chunk range): wipe existing .json/.md
    files and SQLite/Qdrant records for that range, then re-summarize.

    Args:
        args: Parsed command-line arguments.
        config: Loaded configuration dictionary.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger = logging.getLogger(__name__)

    if not hasattr(args, "source") or args.source != "chatgpt":
        logger.error("Only 'chatgpt' source is supported currently.")
        return 1

    conv_dir = Path(args.inbox_dir) / args.conversation_id
    chunks_dir = conv_dir / "chunks"

    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}. Run 'ingest' first.")
        return 1

    output_root = Path(config["output_root"])
    chunk_summaries_dir = output_root / args.conversation_id / "chunk_summaries"
    from_chunk = args.from_chunk if args.from_chunk is not None else 0
    to_chunk = args.to_chunk  # None means "all"

    try:
        # --force: wipe files + DB/Qdrant for the affected range
        if args.force:
            chunk_files = sorted(
                f for f in chunks_dir.glob("chunk_*.json")
                if f.name != "pending_tail.json"
            )
            import json as _json
            affected_chunk_ids = []
            for cf in chunk_files:
                with open(cf, encoding="utf-8") as fh:
                    cd = _json.load(fh)
                idx = cd["chunk_index"]
                effective_to = to_chunk if to_chunk is not None else float("inf")
                if from_chunk <= idx <= effective_to:
                    affected_chunk_ids.append(cd["chunk_id"])

            # Delete files
            for cid in affected_chunk_ids:
                for ext in (".json", ".md"):
                    p = chunk_summaries_dir / f"{cid}{ext}"
                    if p.exists():
                        p.unlink()

            # Delete DB rows + collect Qdrant point IDs
            store = SummaryStore(db_path=config["db_path"])
            point_ids = store.delete_chunk_rows(args.conversation_id, affected_chunk_ids)
            if point_ids:
                try:
                    vs = VectorStore(
                        host=config["qdrant_host"], port=config["qdrant_port"]
                    )
                    vs.delete_points(point_ids)
                except Exception:
                    pass
            logger.info(
                f"--force: wiped {len(affected_chunk_ids)} chunks "
                f"(range [{from_chunk}, {'end' if to_chunk is None else to_chunk}])"
            )

        ollama_client = OllamaClient(
            model=config["local_model_name"],
            base_url=config["ollama_base_url"],
        )
        summarizer = ChunkSummarizer(
            ollama_client=ollama_client,
            prompts_dir=config["prompts_dir"],
            schema=config["schema"],
            schema_version=config["schema_version"],
            context_window=args.context_window,
        )

        logger.info(
            f"Summarizing chunks [{from_chunk}, "
            f"{'end' if to_chunk is None else to_chunk}] "
            f"for conversation {args.conversation_id}"
        )

        results = summarizer.summarize_conversation_chunks(
            chunks_dir=chunks_dir,
            conversation_id=args.conversation_id,
            output_root=output_root,
            from_chunk=from_chunk,
            to_chunk=to_chunk,
        )

        if not results:
            print("No chunks to summarize in the specified range.")
            return 0

        # Count newly summarized vs loaded from disk (latency_ms=0 for disk-loaded)
        new_count = sum(1 for _, d in results if d.get("latency_ms", 0) > 0)
        total_latency = sum(d.get("latency_ms", 0) for _, d in results)

        if args.persist:
            logger.info("Persisting chunk summaries to SQLite + Qdrant...")
            memory = _build_memory_layer(config)
            for output_dir, output_data in results:
                memory.persist(output_data=output_data, output_dir=output_dir)
            logger.info(f"Persisted {len(results)} chunk summaries")

        print(f"\nChunk summarization complete for: {args.conversation_id}")
        print(f"  Chunks processed  : {len(results)} ({new_count} new, "
              f"{len(results) - new_count} from disk)")
        print(f"  Total latency     : {total_latency}ms")
        print(f"  Output directory  : {chunk_summaries_dir}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during chunk summarization: {e}")
        return 1


def cmd_detect_segments(args: argparse.Namespace, config: dict) -> int:
    """Execute the detect-segments command.

    Detects topic boundaries in chunk summaries using cosine similarity and
    summarizes each detected segment with the LLM.

    Args:
        args: Parsed command-line arguments.
        config: Loaded configuration dictionary.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger = logging.getLogger(__name__)

    if not hasattr(args, "source") or args.source != "chatgpt":
        logger.error("Only 'chatgpt' source is supported currently.")
        return 1

    output_root = Path(config["output_root"])
    chunk_summaries_dir = output_root / args.conversation_id / "chunk_summaries"

    if not chunk_summaries_dir.exists():
        logger.error(
            f"Chunk summaries not found: {chunk_summaries_dir}. "
            "Run 'summarize-chunks' first."
        )
        return 1

    try:
        embedder = EmbeddingClient(
            model=config["embedding_model"],
            base_url=config["ollama_base_url"],
        )
        ollama_client = OllamaClient(
            model=config["local_model_name"],
            base_url=config["ollama_base_url"],
        )

        vector_store: Optional[VectorStore] = None
        try:
            vector_store = VectorStore(
                host=config["qdrant_host"],
                port=config["qdrant_port"],
            )
        except Exception:
            logger.info("Qdrant not reachable — will embed on the fly")

        detector = SegmentDetector(
            embedder=embedder,
            ollama_client=ollama_client,
            prompts_dir=config["prompts_dir"],
            schema=config["schema"],
            schema_version=config["schema_version"],
            threshold=args.threshold,
            vector_store=vector_store,
        )

        results = detector.detect_and_summarize(
            conversation_id=args.conversation_id,
            output_root=output_root,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            return 0

        if not results:
            print("No segments produced.")
            return 0

        if args.persist:
            logger.info("Persisting segment summaries to SQLite + Qdrant...")
            memory = _build_memory_layer(config)
            for output_dir, output_data in results:
                memory.persist(output_data=output_data, output_dir=output_dir)
            logger.info(f"Persisted {len(results)} segment summaries")

        segment_summaries_dir = output_root / args.conversation_id / "segment_summaries"
        print(f"\nSegment detection complete for: {args.conversation_id}")
        print(f"  Segments detected : {len(results)}")
        for i, (_, data) in enumerate(results):
            first_sentence = (data.get("summary") or "").split(".")[0]
            chunk_range = data.get("segment_chunk_range", "?")
            print(f"  Segment {i} ({chunk_range}): \"{first_sentence}.\"")
        print(f"  Output directory  : {segment_summaries_dir}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during segment detection: {e}")
        return 1


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="JARVIS - Local-first AI assistant for PM workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- summarize --------------------------------------------------------
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a conversation file"
    )
    summarize_parser.add_argument(
        "--file", "-f", required=True, help="Path to input JSON conversation file"
    )
    summarize_parser.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help="Persist the summary to SQLite and index it in Qdrant",
    )
    summarize_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Wipe existing output files and DB/Qdrant records before re-running",
    )

    # -- retrieve ---------------------------------------------------------
    retrieve_parser = subparsers.add_parser(
        "retrieve", help="Semantic search over persisted summaries"
    )
    retrieve_parser.add_argument(
        "--query", "-q", required=True, help="Natural-language search query"
    )
    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results to return (default: 5)",
    )

    # -- ingest -----------------------------------------------------------
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest and normalize a raw conversation export"
    )
    ingest_sub = ingest_parser.add_subparsers(dest="source", help="Source platform")

    ingest_chatgpt = ingest_sub.add_parser(
        "chatgpt", help="Ingest a raw ChatGPT conversation export"
    )
    ingest_chatgpt.add_argument(
        "--file", "-f", required=True, help="Path to raw ChatGPT export JSON"
    )
    ingest_chatgpt.add_argument(
        "--output-dir",
        default="inbox/ai_chat/chatgpt",
        help="Base output directory (default: inbox/ai_chat/chatgpt)",
    )

    # -- summarize-chunks -------------------------------------------------
    sc_parser = subparsers.add_parser(
        "summarize-chunks", help="Summarize all chunks of an ingested conversation"
    )
    sc_sub = sc_parser.add_subparsers(dest="source", help="Source platform")

    sc_chatgpt = sc_sub.add_parser(
        "chatgpt", help="Summarize chunks from a ChatGPT ingested conversation"
    )
    sc_chatgpt.add_argument(
        "--conversation-id", required=True, dest="conversation_id",
        help="Conversation ID (subfolder name under --inbox-dir)",
    )
    sc_chatgpt.add_argument(
        "--inbox-dir",
        default="inbox/ai_chat/chatgpt",
        dest="inbox_dir",
        help="Base inbox directory (default: inbox/ai_chat/chatgpt)",
    )
    sc_chatgpt.add_argument(
        "--from-chunk",
        type=int,
        default=None,
        dest="from_chunk",
        help="Start at this chunk index, inclusive (default: 0)",
    )
    sc_chatgpt.add_argument(
        "--to-chunk",
        type=int,
        default=None,
        dest="to_chunk",
        help="Stop after this chunk index, inclusive (default: last)",
    )
    sc_chatgpt.add_argument(
        "--context-window",
        type=int,
        default=3,
        dest="context_window",
        help="Number of prior chunk summaries to pass as context (default: 3)",
    )
    sc_chatgpt.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help="Persist summaries to SQLite and index in Qdrant",
    )
    sc_chatgpt.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Wipe existing files and DB/Qdrant records for the range before re-running",
    )

    # -- detect-segments --------------------------------------------------
    ds_parser = subparsers.add_parser(
        "detect-segments", help="Detect topic segments and summarize each one"
    )
    ds_sub = ds_parser.add_subparsers(dest="source", help="Source platform")

    ds_chatgpt = ds_sub.add_parser(
        "chatgpt", help="Detect segments from a ChatGPT ingested conversation"
    )
    ds_chatgpt.add_argument(
        "--conversation-id", required=True, dest="conversation_id",
        help="Conversation ID (used to locate OUTPUTS/<id>/chunk_summaries/)",
    )
    ds_chatgpt.add_argument(
        "--inbox-dir",
        default="inbox/ai_chat/chatgpt",
        dest="inbox_dir",
        help="Base inbox directory (default: inbox/ai_chat/chatgpt)",
    )
    ds_chatgpt.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Cosine similarity drop threshold for segment boundaries (default: 0.65)",
    )
    ds_chatgpt.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Detect boundaries only — skip LLM summarization",
    )
    ds_chatgpt.add_argument(
        "--persist",
        action="store_true",
        default=False,
        help="Persist segment summaries to SQLite and index in Qdrant",
    )

    # -- parse & dispatch -------------------------------------------------
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    config = load_config()
    setup_logging(config["log_level"])

    if args.command == "summarize":
        return cmd_summarize(args, config)
    if args.command == "retrieve":
        return cmd_retrieve(args, config)
    if args.command == "ingest":
        return cmd_ingest(args, config)
    if args.command == "summarize-chunks":
        return cmd_summarize_chunks(args, config)
    if args.command == "detect-segments":
        return cmd_detect_segments(args, config)

    return 1


if __name__ == "__main__":
    sys.exit(main())
