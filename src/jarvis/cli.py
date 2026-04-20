"""CLI entry point for JARVIS."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from jarvis.segment_summarizer import SegmentSummarizer
from jarvis.config import load_config
from jarvis.extractor import SegmentExtractor
from jarvis.fragmenter import Fragmenter
from jarvis.topic_detector import TopicDetector
from jarvis.embedder import EmbeddingClient
from jarvis.ingest.chatgpt_parser import load_raw_export, parse_export
from jarvis.ingest.segmenter import segment_conversation, save_segments
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
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_memory_layer(config: dict) -> MemoryLayer:
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
    """Execute the summarize command."""
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
            timeout=config["ollama_timeout"],
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
    """Execute the retrieve command."""
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


def _build_context_block(rows: list[dict]) -> str:
    parts = []
    for i, row in enumerate(rows, start=1):
        bullets = row.get("bullets") or []
        bullet_text = "\n".join(f"• {b}" for b in bullets) if bullets else ""
        header = f"--- Excerpt {i} (source: {row['source_file']}, date: {row['created_at']}) ---"
        body = row.get("summary") or ""
        if bullet_text:
            body += f"\nKey points:\n{bullet_text}"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def cmd_answer(args: argparse.Namespace, config: dict) -> int:
    """Execute the answer command."""
    logger = logging.getLogger(__name__)
    try:
        memory = _build_memory_layer(config)
        ollama = OllamaClient(
            base_url=config["ollama_base_url"],
            model=config["local_model_name"],
            timeout=config["ollama_timeout"],
        )

        logger.info(f"Embedding query with model={config['embedding_model']}...")
        query_vector = memory.embedder.embed(args.query)
        hits = memory.vector_store.search(query_vector=query_vector, top_k=args.top_k)

        if not hits:
            print("No relevant context found for your question.")
            return 0

        summary_ids = [sid for sid, _, _ in hits]
        rows = memory.store.get_by_ids(summary_ids)

        context_block = _build_context_block(rows)

        prompt_path = Path(config["prompts_dir"]) / "answer_question.md"
        template = prompt_path.read_text(encoding="utf-8")
        prompt = template.replace("{question}", args.query).replace("{context_block}", context_block)

        logger.info(f"Generating answer with model={config['local_model_name']}...")
        raw, is_degraded, warning = ollama.generate(prompt, temperature=args.temperature)

        if is_degraded:
            logger.warning(f"Degraded response: {warning}")

        answer = raw.strip()
        if "## Answer" in answer:
            answer = answer.split("## Answer", 1)[-1].strip()

        print(f"\nAnswer to: \"{args.query}\"\n")
        print("─" * 72)
        print(answer)
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
    """Execute the ingest command."""
    logger = logging.getLogger(__name__)

    if not hasattr(args, "source") or args.source != "chatgpt":
        logger.error("Only 'chatgpt' source is supported currently.")
        return 1

    raw_path = Path(args.file)
    if not raw_path.exists():
        logger.error(f"Raw export file not found: {raw_path}")
        return 1

    try:
        logger.info(f"Loading raw export: {raw_path}")
        raw = load_raw_export(str(raw_path))
        messages = parse_export(raw)
        logger.info(f"Parsed {len(messages)} visible messages")

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

        segments_dir = conv_dir / "segments"
        result = segment_conversation(normalized)
        save_segments(
            segments=result["segments"],
            pending_tail=result["pending_tail"],
            manifest_meta=result["manifest_meta"],
            output_dir=segments_dir,
        )

        print(f"\nIngest complete for: {normalized.get('title', conversation_id)}")
        print(f"  Visible messages  : {normalized['message_count']}")
        print(f"  Segments written  : {len(result['segments'])}")
        if result["pending_tail"]:
            print("  Pending tail      : 1 unmatched trailing user message")
        print(f"  Normalized file   : {norm_path}")
        print(f"  Segments directory: {segments_dir}")
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


def cmd_summarize_segments(args: argparse.Namespace, config: dict) -> int:
    """Execute the summarize-segments command."""
    logger = logging.getLogger(__name__)

    if not hasattr(args, "source") or args.source != "chatgpt":
        logger.error("Only 'chatgpt' source is supported currently.")
        return 1

    conv_dir = Path(args.inbox_dir) / args.conversation_id
    segments_dir = conv_dir / "segments"

    if not segments_dir.exists():
        logger.error(f"Segments directory not found: {segments_dir}. Run 'ingest' first.")
        return 1

    output_root = Path(config["output_root"])
    segment_summaries_dir = output_root / args.conversation_id / "segment_summaries"
    from_segment = args.from_segment if args.from_segment is not None else 0
    to_segment = args.to_segment

    try:
        if args.force:
            segment_files = sorted(
                f for f in segments_dir.glob("segment_*.json")
                if f.name != "pending_tail.json"
            )
            import json as _json
            affected_segment_ids = []
            for sf in segment_files:
                with open(sf, encoding="utf-8") as fh:
                    sd = _json.load(fh)
                idx = sd["segment_index"]
                effective_to = to_segment if to_segment is not None else float("inf")
                if from_segment <= idx <= effective_to:
                    affected_segment_ids.append(sd["segment_id"])

            for sid in affected_segment_ids:
                for ext in (".json", ".md"):
                    p = segment_summaries_dir / f"{sid}{ext}"
                    if p.exists():
                        p.unlink()

            store = SummaryStore(db_path=config["db_path"])
            point_ids = store.delete_segment_rows(args.conversation_id, affected_segment_ids)
            if point_ids:
                try:
                    vs = VectorStore(
                        host=config["qdrant_host"], port=config["qdrant_port"]
                    )
                    vs.delete_points(point_ids)
                except Exception:
                    pass
            logger.info(
                f"--force: wiped {len(affected_segment_ids)} segments "
                f"(range [{from_segment}, {'end' if to_segment is None else to_segment}])"
            )

        ollama_client = OllamaClient(
            model=config["local_model_name"],
            base_url=config["ollama_base_url"],
            timeout=config["ollama_timeout"],
        )
        summarizer = SegmentSummarizer(
            ollama_client=ollama_client,
            prompts_dir=config["prompts_dir"],
            schema=config["schema"],
            schema_version=config["schema_version"],
            context_window=args.context_window,
        )

        logger.info(
            f"Summarizing segments [{from_segment}, "
            f"{'end' if to_segment is None else to_segment}] "
            f"for conversation {args.conversation_id}"
        )

        results = summarizer.summarize_conversation_segments(
            segments_dir=segments_dir,
            conversation_id=args.conversation_id,
            output_root=output_root,
            from_segment=from_segment,
            to_segment=to_segment,
        )

        if not results:
            print("No segments to summarize in the specified range.")
            return 0

        new_count = sum(1 for _, d in results if d.get("latency_ms", 0) > 0)
        total_latency = sum(d.get("latency_ms", 0) for _, d in results)

        if args.persist:
            logger.info("Persisting segment summaries to SQLite + Qdrant...")
            memory = _build_memory_layer(config)
            for output_dir, output_data in results:
                memory.persist(output_data=output_data, output_dir=output_dir)
            logger.info(f"Persisted {len(results)} segment summaries")

        print(f"\nSegment summarization complete for: {args.conversation_id}")
        print(f"  Segments processed: {len(results)} ({new_count} new, "
              f"{len(results) - new_count} from disk)")
        print(f"  Total latency     : {total_latency}ms")
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
        logger.exception(f"Unexpected error during segment summarization: {e}")
        return 1


def cmd_detect_topics(args: argparse.Namespace, config: dict) -> int:
    """Execute the detect-topics command."""
    logger = logging.getLogger(__name__)

    if not hasattr(args, "source") or args.source != "chatgpt":
        logger.error("Only 'chatgpt' source is supported currently.")
        return 1

    output_root = Path(config["output_root"])
    segment_summaries_dir = output_root / args.conversation_id / "segment_summaries"

    if not segment_summaries_dir.exists():
        logger.error(
            f"Segment summaries not found: {segment_summaries_dir}. "
            "Run 'summarize-segments' first."
        )
        return 1

    topic_summaries_dir = output_root / args.conversation_id / "topic_summaries"

    try:
        if args.force:
            store = SummaryStore(db_path=config["db_path"])
            point_ids = store.delete_topic_rows(args.conversation_id)
            if point_ids:
                try:
                    vs = VectorStore(
                        host=config["qdrant_host"], port=config["qdrant_port"]
                    )
                    vs.delete_points(point_ids)
                except Exception:
                    pass
            if topic_summaries_dir.exists():
                for f in topic_summaries_dir.glob("topic_*"):
                    f.unlink()
            logger.info(
                f"--force: wiped existing topic summaries for {args.conversation_id}"
            )

        embedder = EmbeddingClient(
            model=config["embedding_model"],
            base_url=config["ollama_base_url"],
        )
        ollama_client = OllamaClient(
            model=config["local_model_name"],
            base_url=config["ollama_base_url"],
            timeout=config["ollama_timeout"],
        )

        vector_store: Optional[VectorStore] = None
        try:
            vector_store = VectorStore(
                host=config["qdrant_host"],
                port=config["qdrant_port"],
            )
        except Exception:
            logger.info("Qdrant not reachable — will embed on the fly")

        detector = TopicDetector(
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
            print("No topics produced.")
            return 0

        if args.persist:
            logger.info("Persisting topic summaries to SQLite + Qdrant...")
            memory = _build_memory_layer(config)
            for output_dir, output_data in results:
                memory.persist(output_data=output_data, output_dir=output_dir)
            logger.info(f"Persisted {len(results)} topic summaries")

        print(f"\nTopic detection complete for: {args.conversation_id}")
        print(f"  Topics detected   : {len(results)}")
        for i, (_, data) in enumerate(results):
            first_sentence = (data.get("summary") or "").split(".")[0]
            seg_range = data.get("topic_segment_range", "?")
            print(f"  Topic {i} ({seg_range}): \"{first_sentence}.\"")
        print(f"  Output directory  : {topic_summaries_dir}")
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
        logger.exception(f"Unexpected error during topic detection: {e}")
        return 1


def cmd_extract_segments(args: argparse.Namespace, config: dict) -> int:
    """Execute the extract-segments command."""
    logger = logging.getLogger(__name__)

    if not hasattr(args, "source") or args.source != "chatgpt":
        logger.error("Only 'chatgpt' source is supported currently.")
        return 1

    conv_dir = Path(args.inbox_dir) / args.conversation_id
    segments_dir = conv_dir / "segments"

    if not segments_dir.exists():
        logger.error(f"Segments directory not found: {segments_dir}. Run 'ingest' first.")
        return 1

    output_root = Path(config["output_root"])
    extract_dir = output_root / args.conversation_id / "extracts"
    from_segment = args.from_segment if args.from_segment is not None else 0
    to_segment = args.to_segment

    try:
        if args.force:
            store = SummaryStore(db_path=config["db_path"])
            point_ids = store.delete_extract_rows(args.conversation_id)
            if point_ids:
                try:
                    vs = VectorStore(host=config["qdrant_host"], port=config["qdrant_port"])
                    vs.delete_points(point_ids)
                except Exception:
                    pass
            if extract_dir.exists():
                for f in extract_dir.glob("extract_*"):
                    f.unlink()
            logger.info(
                f"--force: wiped existing extracts for {args.conversation_id}"
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
            conversation_id=args.conversation_id,
            output_root=output_root,
            from_segment=from_segment,
            to_segment=to_segment,
            force=args.force,
        )

        if not results:
            print("No segments to extract in the specified range.")
            return 0

        new_count = sum(1 for _, d in results if d.get("latency_ms", 0) > 0)

        if args.persist:
            logger.info("Persisting extracts to SQLite + Qdrant...")
            memory = _build_memory_layer(config)
            for output_dir, output_data in results:
                memory.persist(output_data=output_data, output_dir=output_dir)
            logger.info(f"Persisted {len(results)} extracts")

        print(f"\nExtraction complete for: {args.conversation_id}")
        print(f"  Segments processed: {len(results)} ({new_count} new, "
              f"{len(results) - new_count} from disk)")
        print(f"  Output directory  : {extract_dir}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during extraction: {e}")
        return 1


def cmd_fragment_extracts(args: argparse.Namespace, config: dict) -> int:
    """Execute the fragment-extracts command."""
    logger = logging.getLogger(__name__)

    if not hasattr(args, "source") or args.source != "chatgpt":
        logger.error("Only 'chatgpt' source is supported currently.")
        return 1

    output_root = Path(config["output_root"])
    extract_dir = output_root / args.conversation_id / "extracts"

    if not extract_dir.exists():
        logger.error(
            f"Extracts directory not found: {extract_dir}. "
            "Run 'extract-segments' first."
        )
        return 1

    fragment_dir = output_root / args.conversation_id / "fragments"

    try:
        if args.force:
            store = SummaryStore(db_path=config["db_path"])
            point_ids = store.delete_fragment_rows(args.conversation_id)
            if point_ids:
                try:
                    vs = VectorStore(host=config["qdrant_host"], port=config["qdrant_port"])
                    vs.delete_points(point_ids)
                except Exception:
                    pass
            if fragment_dir.exists():
                for f in fragment_dir.glob("fragment_*"):
                    f.unlink()
            logger.info(
                f"--force: wiped existing fragments for {args.conversation_id}"
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

        from_segment = args.from_segment if args.from_segment is not None else 0
        to_segment = args.to_segment

        results = fragmenter.fragment_conversation_extracts(
            conversation_id=args.conversation_id,
            output_root=output_root,
            from_segment=from_segment,
            to_segment=to_segment,
            force=args.force,
        )

        if not results:
            print("No fragments produced.")
            return 0

        if args.persist:
            logger.info("Persisting fragments to SQLite + Qdrant...")
            memory = _build_memory_layer(config)
            for output_dir, output_data in results:
                memory.persist(output_data=output_data, output_dir=output_dir)
            logger.info(f"Persisted {len(results)} fragments")

        print(f"\nFragmentation complete for: {args.conversation_id}")
        print(f"  Fragments produced: {len(results)}")
        print(f"  Output directory  : {fragment_dir}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during fragmentation: {e}")
        return 1


def main() -> int:
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
        "--persist", action="store_true", default=False,
        help="Persist the summary to SQLite and index it in Qdrant",
    )
    summarize_parser.add_argument(
        "--force", action="store_true", default=False,
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
        "--top-k", type=int, default=5, dest="top_k",
        help="Number of results to return (default: 5)",
    )

    # -- answer -----------------------------------------------------------
    answer_parser = subparsers.add_parser(
        "answer", help="Answer a question using indexed data"
    )
    answer_parser.add_argument("query", help="Natural-language question")
    answer_parser.add_argument(
        "--top-k", type=int, default=5, dest="top_k",
        help="Number of context excerpts to retrieve (default: 5)",
    )
    answer_parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM sampling temperature (default: 0.3)",
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
        "--output-dir", default="inbox/ai_chat/chatgpt",
        help="Base output directory (default: inbox/ai_chat/chatgpt)",
    )

    # -- summarize-segments -----------------------------------------------
    ss_parser = subparsers.add_parser(
        "summarize-segments", help="Summarize all segments of an ingested conversation"
    )
    ss_sub = ss_parser.add_subparsers(dest="source", help="Source platform")

    ss_chatgpt = ss_sub.add_parser(
        "chatgpt", help="Summarize segments from a ChatGPT ingested conversation"
    )
    ss_chatgpt.add_argument(
        "--conversation-id", required=True, dest="conversation_id",
        help="Conversation ID (subfolder name under --inbox-dir)",
    )
    ss_chatgpt.add_argument(
        "--inbox-dir", default="inbox/ai_chat/chatgpt", dest="inbox_dir",
        help="Base inbox directory (default: inbox/ai_chat/chatgpt)",
    )
    ss_chatgpt.add_argument(
        "--from-segment", type=int, default=None, dest="from_segment",
        help="Start at this segment index, inclusive (default: 0)",
    )
    ss_chatgpt.add_argument(
        "--to-segment", type=int, default=None, dest="to_segment",
        help="Stop after this segment index, inclusive (default: last)",
    )
    ss_chatgpt.add_argument(
        "--context-window", type=int, default=3, dest="context_window",
        help="Number of prior segment summaries to pass as context (default: 3)",
    )
    ss_chatgpt.add_argument(
        "--persist", action="store_true", default=False,
        help="Persist summaries to SQLite and index in Qdrant",
    )
    ss_chatgpt.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing files and DB/Qdrant records for the range before re-running",
    )

    # -- detect-topics ----------------------------------------------------
    dt_parser = subparsers.add_parser(
        "detect-topics", help="Detect topics across segments and summarize each one"
    )
    dt_sub = dt_parser.add_subparsers(dest="source", help="Source platform")

    dt_chatgpt = dt_sub.add_parser(
        "chatgpt", help="Detect topics from a ChatGPT ingested conversation"
    )
    dt_chatgpt.add_argument(
        "--conversation-id", required=True, dest="conversation_id",
        help="Conversation ID (used to locate OUTPUTS/<id>/segment_summaries/)",
    )
    dt_chatgpt.add_argument(
        "--inbox-dir", default="inbox/ai_chat/chatgpt", dest="inbox_dir",
        help="Base inbox directory (default: inbox/ai_chat/chatgpt)",
    )
    dt_chatgpt.add_argument(
        "--threshold", type=float, default=0.55,
        help="Cosine similarity drop threshold for topic boundaries (default: 0.55)",
    )
    dt_chatgpt.add_argument(
        "--dry-run", action="store_true", default=False, dest="dry_run",
        help="Detect boundaries only — skip LLM summarization",
    )
    dt_chatgpt.add_argument(
        "--persist", action="store_true", default=False,
        help="Persist topic summaries to SQLite and index in Qdrant",
    )
    dt_chatgpt.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing topic files and DB/Qdrant records before re-running",
    )

    # -- extract-segments -------------------------------------------------
    es_parser = subparsers.add_parser(
        "extract-segments", help="Extract attributed statements from each segment"
    )
    es_sub = es_parser.add_subparsers(dest="source", help="Source platform")

    es_chatgpt = es_sub.add_parser(
        "chatgpt", help="Extract from a ChatGPT ingested conversation"
    )
    es_chatgpt.add_argument(
        "--conversation-id", required=True, dest="conversation_id",
        help="Conversation ID (subfolder under --inbox-dir)",
    )
    es_chatgpt.add_argument(
        "--inbox-dir", default="inbox/ai_chat/chatgpt", dest="inbox_dir",
        help="Base inbox directory (default: inbox/ai_chat/chatgpt)",
    )
    es_chatgpt.add_argument(
        "--from-segment", type=int, default=None, dest="from_segment",
        help="Start at this segment index, inclusive (default: 0)",
    )
    es_chatgpt.add_argument(
        "--to-segment", type=int, default=None, dest="to_segment",
        help="Stop after this segment index, inclusive (default: last)",
    )
    es_chatgpt.add_argument(
        "--persist", action="store_true", default=False,
        help="Persist extracts to SQLite and index in Qdrant",
    )
    es_chatgpt.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing extract files and records before re-running",
    )

    # -- fragment-extracts ------------------------------------------------
    fe_parser = subparsers.add_parser(
        "fragment-extracts", help="Fragment extracted statements into retrieval units"
    )
    fe_sub = fe_parser.add_subparsers(dest="source", help="Source platform")

    fe_chatgpt = fe_sub.add_parser(
        "chatgpt", help="Fragment extracts from a ChatGPT ingested conversation"
    )
    fe_chatgpt.add_argument(
        "--conversation-id", required=True, dest="conversation_id",
        help="Conversation ID",
    )
    fe_chatgpt.add_argument(
        "--from-segment", type=int, default=None, dest="from_segment",
        help="Start at this segment index, inclusive (default: 0)",
    )
    fe_chatgpt.add_argument(
        "--to-segment", type=int, default=None, dest="to_segment",
        help="Stop after this segment index, inclusive (default: last)",
    )
    fe_chatgpt.add_argument(
        "--persist", action="store_true", default=False,
        help="Persist fragments to SQLite and index in Qdrant",
    )
    fe_chatgpt.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing fragment files and records before re-running",
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
    if args.command == "answer":
        return cmd_answer(args, config)
    if args.command == "ingest":
        return cmd_ingest(args, config)
    if args.command == "summarize-segments":
        return cmd_summarize_segments(args, config)
    if args.command == "detect-topics":
        return cmd_detect_topics(args, config)
    if args.command == "extract-segments":
        return cmd_extract_segments(args, config)
    if args.command == "fragment-extracts":
        return cmd_fragment_extracts(args, config)

    return 1


if __name__ == "__main__":
    sys.exit(main())
