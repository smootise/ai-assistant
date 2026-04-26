"""CLI entry point for JARVIS."""

import argparse
import json
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
from jarvis.ingest.pipeline import ingest_chatgpt
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
            for p in (json_path, md_path):
                if p.exists():
                    p.unlink()
            logger.info(f"--force: wiped existing outputs for {source_basename}")
        elif json_path.exists():
            pass  # OutputWriter will overwrite; no DB record to clean up

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
            logger.warning(
                "--persist is not yet implemented for cmd_summarize. "
                "Use the chatgpt pipeline (ingest → extract-segments → fragment-extracts)."
            )

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


def _apply_hybrid_cutoff(
    hits: list, min_results: int, min_score: float
) -> list:
    """Return at least min_results hits, then continue while score >= min_score."""
    result = []
    for i, hit in enumerate(hits):
        _, score, _ = hit
        if i < min_results or score >= min_score:
            result.append(hit)
        else:
            break
    return result


def cmd_retrieve(args: argparse.Namespace, config: dict) -> int:
    """Execute the retrieve command."""
    logger = logging.getLogger(__name__)

    try:
        memory = _build_memory_layer(config)

        logger.info(f"Embedding query with model={config['embedding_model']}...")
        query_vector = memory.embedder.embed(args.query)

        hits = memory.vector_store.search(query_vector=query_vector, top_k=args.top_k)
        hits = _apply_hybrid_cutoff(hits, args.min_results, args.min_score)

        if not hits:
            print("No results found.")
            return 0

        fragment_ids = [fid for fid, _, _ in hits]
        scores = {fid: score for fid, score, _ in hits}
        rows = memory.store.get_fragments_with_statements(fragment_ids)

        print(f"\nTop {len(rows)} result(s) for: \"{args.query}\"\n")
        print("-" * 72)

        for rank, row in enumerate(rows, start=1):
            fid = row["fragment_id"]
            score = scores.get(fid, 0.0)

            title = row.get("title") or ""
            stmts = row.get("statements", [])
            body = " / ".join(
                f"{s['speaker']}: {s['text']}" for s in stmts[:2]
            )
            raw_preview = f"[{title}] {body}" if title else body
            preview = raw_preview[:160].replace("\n", " ")
            if len(raw_preview) > 160:
                preview += "..."

            print(f"#{rank}  score={score:.4f}  fragment={fid}")
            print(f"    segment: {row.get('segment_id', '?')}")
            print(f"    date   : {row.get('conversation_date', '?')}")
            print(f"    preview: {preview}")
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
        date = row.get("conversation_date") or row.get("fragment_created_at", "")
        segment_id = row.get("segment_id", "?")
        header = f"--- Excerpt {i} (segment: {segment_id}, date: {date}) ---"
        title = row.get("title") or ""
        lines = []
        if title:
            lines.append(f"Topic: {title}")
        for s in row.get("statements", []):
            lines.append(f"{s['speaker']}: {s['text']}")
        body = "\n".join(lines) if lines else row.get("retrieval_text", "")
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
        hits = _apply_hybrid_cutoff(hits, args.min_results, args.min_score)

        if not hits:
            print("No relevant context found for your question.")
            return 0

        fragment_ids = [fid for fid, _, _ in hits]
        rows = memory.store.get_fragments_with_statements(fragment_ids)

        context_block = _build_context_block(rows)

        user_name = config.get("user_name", "").strip()
        if user_name:
            user_context = (
                f"The person who owns this assistant is {user_name}. "
                f"In the context excerpts, 'user' and '{user_name}' refer to the same person — "
                f"the one speaking or being spoken about."
            )
        else:
            user_context = (
                "In the context excerpts, 'user' refers to the person who owns this assistant."
            )

        prompt_path = Path(config["prompts_dir"]) / "answer_question.md"
        template = prompt_path.read_text(encoding="utf-8")
        system_prompt = (
            template
            .replace("{question}", args.query)
            .replace("{context_block}", context_block)
            .replace("{user_context}", user_context)
        )

        if args.verbose:
            print(f"\nContext excerpts for: \"{args.query}\"\n")
            print("-" * 72)
            print(context_block)
            print("-" * 72)
            print()

        logger.info(f"Generating answer with model={config['local_model_name']}...")
        raw, is_degraded, warning = ollama.chat(
            system_prompt, args.query, temperature=args.temperature
        )

        if is_degraded:
            logger.warning(f"Degraded response: {warning}")

        answer = raw.strip()
        if "## Answer" in answer:
            answer = answer.split("## Answer", 1)[-1].strip()

        print(f"\nAnswer to: \"{args.query}\"\n")
        print("-" * 72)
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

    try:
        result = ingest_chatgpt(
            raw_path=Path(args.file),
            output_dir=Path(args.output_dir),
            persist=getattr(args, "persist", False),
            config=config,
        )
        conv_dir = Path(args.output_dir) / result["conversation_id"]
        print(f"\nIngest complete for: {result['conversation_id']}")
        print(f"  Segments written  : {result['segment_count']}")
        print(f"  Normalized file   : {result['normalized_path']}")
        print(f"  Segments directory: {conv_dir / 'segments'}")
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
            affected_segment_ids = []
            for sf in segment_files:
                with open(sf, encoding="utf-8") as fh:
                    sd = json.load(fh)
                idx = sd["segment_index"]
                effective_to = to_segment if to_segment is not None else float("inf")
                if from_segment <= idx <= effective_to:
                    affected_segment_ids.append(sd["segment_id"])

            for sid in affected_segment_ids:
                for ext in (".json", ".md"):
                    p = segment_summaries_dir / f"{sid}{ext}"
                    if p.exists():
                        p.unlink()

            if args.persist:
                store = SummaryStore(db_path=config["db_path"])
                store.delete_segment_summaries(args.conversation_id)

            logger.info(
                f"--force: wiped {len(affected_segment_ids)} segment summaries "
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

        new_count = sum(1 for _, d in results if not d.get("_from_disk"))
        total_latency = sum(d.get("latency_ms", 0) for _, d in results)

        if args.persist:
            logger.info("Persisting segment summaries to SQLite...")
            memory = _build_memory_layer(config)
            persisted = 0
            for _, output_data in results:
                segment_id = output_data.get("segment_id")
                if not segment_id:
                    continue
                existing = memory.store.get_segment_summary(segment_id)
                if existing:
                    logger.debug(f"Skipping already-persisted summary: {segment_id}")
                    continue
                memory.persist_segment_summary(output_data)
                persisted += 1
            logger.info(
                f"Persisted {persisted} segment summaries "
                f"({len(results) - persisted} already existed)"
            )

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
            if topic_summaries_dir.exists():
                for f in topic_summaries_dir.glob("topic_*"):
                    f.unlink()
            if args.persist:
                store = SummaryStore(db_path=config["db_path"])
                store.delete_topic_summaries(args.conversation_id)
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
            logger.info("Persisting topic summaries to SQLite...")
            memory = _build_memory_layer(config)
            persisted = 0
            for _, output_data in results:
                topic_idx = output_data.get("topic_index")
                if topic_idx is None:
                    continue
                conv_id = output_data.get("parent_conversation_id", args.conversation_id)
                # Check idempotency using topic_id = <conv_id>_t<NNN>
                topic_id = f"{conv_id}_t{topic_idx:03d}"
                with memory.store._connect() as conn:
                    existing = conn.execute(
                        "SELECT topic_id FROM topic_summaries WHERE topic_id = ?",
                        (topic_id,),
                    ).fetchone()
                if existing:
                    logger.debug(f"Skipping already-persisted topic: {topic_id}")
                    continue
                # Collect segment_ids from the topic's segment range
                segment_ids: list = []
                seg_range = output_data.get("topic_segment_range", "")
                if seg_range:
                    try:
                        parts_range = seg_range.replace("s", "").split("-")
                        if len(parts_range) == 2:
                            start_idx, end_idx = int(parts_range[0]), int(parts_range[1])
                            for idx in range(start_idx, end_idx + 1):
                                segment_ids.append(f"{conv_id}_s{idx:03d}")
                    except (ValueError, IndexError):
                        pass
                memory.persist_topic_summary(output_data, segment_ids=segment_ids or None)
                persisted += 1
            logger.info(
                f"Persisted {persisted} topic summaries "
                f"({len(results) - persisted} already existed)"
            )

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
            all_seg_indices = sorted(
                int(f.stem.split("_")[1])
                for f in segments_dir.glob("segment_*.json")
                if f.name != "pending_tail.json"
            )
            effective_from = from_segment
            effective_to = (
                to_segment if to_segment is not None
                else (max(all_seg_indices) if all_seg_indices else from_segment)
            )
            forced_indices = list(range(effective_from, effective_to + 1))

            if extract_dir.exists():
                for idx in forced_indices:
                    for suffix in (".json", ".md"):
                        p = extract_dir / f"extract_{idx:03d}{suffix}"
                        if p.exists():
                            p.unlink()

            if getattr(args, "persist", False):
                store = SummaryStore(db_path=config["db_path"])
                point_ids = store.delete_extracts(
                    args.conversation_id, segment_indices=forced_indices
                )
                if point_ids:
                    try:
                        vs = VectorStore(host=config["qdrant_host"], port=config["qdrant_port"])
                        vs.delete_points(point_ids)
                    except Exception:
                        pass

            logger.info(
                f"--force: wiped extracts {effective_from}–{effective_to} "
                f"for {args.conversation_id}"
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
            retries=getattr(args, "retries", 1),
        )

        if not results:
            print("No segments to extract in the specified range.")
            return 0

        new_count = sum(1 for _, d in results if not d.get("_from_disk"))
        skipped = [
            (d["segment_id"], d.get("warnings", []))
            for _, d in results if d.get("status") == "skipped"
        ]

        if getattr(args, "persist", False):
            logger.info("Persisting extracts to SQLite...")
            memory = _build_memory_layer(config)
            persisted = 0
            for _, output_data in results:
                segment_id = output_data.get("segment_id")
                if not segment_id:
                    continue
                existing = memory.store.get_extract_by_segment(segment_id)
                if existing:
                    logger.debug(f"Skipping already-persisted extract: {segment_id}")
                    continue
                memory.persist_extract_with_statements(output_data)
                persisted += 1
            logger.info(
                f"Persisted {persisted} extracts "
                f"({len(results) - persisted} already existed)"
            )

        print(f"\nExtraction complete for: {args.conversation_id}")
        print(f"  Segments processed: {len(results)} ({new_count} new, "
              f"{len(results) - new_count} from disk)")
        if skipped:
            print(f"  Skipped           : {len(skipped)}")
            for seg_id, warnings in skipped:
                reason = warnings[0] if warnings else "unknown"
                print(f"    - {seg_id}: {reason}")
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
            point_ids = store.delete_fragments(args.conversation_id)
            if point_ids:
                try:
                    vs = VectorStore(host=config["qdrant_host"], port=config["qdrant_port"])
                    vs.delete_points(point_ids)
                except Exception:
                    pass
            if fragment_dir.exists():
                import shutil
                for seg_dir in fragment_dir.glob("segment_*"):
                    if seg_dir.is_dir():
                        shutil.rmtree(seg_dir)
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

        results, skipped_segments = fragmenter.fragment_conversation_extracts(
            conversation_id=args.conversation_id,
            output_root=output_root,
            from_segment=from_segment,
            to_segment=to_segment,
            force=args.force,
            retries=getattr(args, "retries", 1),
        )

        if not results and not skipped_segments:
            print("No fragments produced.")
            return 0

        if args.persist:
            if args.embed:
                logger.info("Persisting fragments to SQLite and indexing in Qdrant...")
            else:
                logger.info("Persisting fragments to SQLite...")
            memory = _build_memory_layer(config)
            persisted = 0
            for _, output_data in results:
                segment_id = output_data.get("segment_id")
                frag_idx = output_data.get("fragment_index")
                if segment_id is None or frag_idx is None:
                    continue

                # Derive extract_id deterministically
                extract_id = f"{segment_id}_x"
                fragment_id = f"{extract_id}_f{frag_idx:03d}"

                existing = memory.store.get_fragment(fragment_id)
                if existing:
                    # Repair missing statement links (can happen if --persist ran before
                    # extract-segments populated extract_statements, or when loading
                    # pre-refactor disk fragments that lack statement_index fields).
                    link_count = memory.store.get_link_count(fragment_id)
                    if link_count == 0 and output_data.get("statements"):
                        frag_stmts = output_data["statements"]
                        # Try deterministic path first (statement_index present)
                        if isinstance(frag_stmts[0].get("statement_index"), int):
                            statement_ids = [
                                f"{extract_id}_st{s['statement_index']:04d}"
                                for s in frag_stmts
                            ]
                        else:
                            # Fall back: match by text against SQLite extract_statements
                            db_stmts = memory.store.get_statements_for_extract(extract_id)
                            text_to_id = {s["text"]: s["statement_id"] for s in db_stmts}
                            statement_ids = [
                                text_to_id[s["text"]]
                                for s in frag_stmts
                                if s["text"] in text_to_id
                            ]
                        if statement_ids:
                            memory.store.insert_fragment_links(fragment_id, statement_ids)
                            logger.debug(
                                f"Repaired {len(statement_ids)} links for {fragment_id}"
                            )

                    # Check Qdrant status
                    if args.embed and not existing.get("qdrant_point_id"):
                        logger.debug(
                            f"Already in SQLite, indexing in Qdrant: {fragment_id}"
                        )
                        memory.index_fragment_in_qdrant(
                            fragment_id=fragment_id, output_data=output_data
                        )
                        persisted += 1
                    else:
                        logger.debug(f"Skipping already-persisted: {fragment_id}")
                    continue

                frag_id = memory.persist_fragment_with_links(
                    output_data=output_data, extract_id=extract_id
                )
                if args.embed:
                    memory.index_fragment_in_qdrant(
                        fragment_id=frag_id, output_data=output_data
                    )
                persisted += 1

            skipped_count = len(results) - persisted
            logger.info(f"Persisted {persisted} fragments ({skipped_count} already existed)")

        print(f"\nFragmentation complete for: {args.conversation_id}")
        print(f"  Fragments produced: {len(results)}")
        if skipped_segments:
            print(f"  Skipped segments  : {len(skipped_segments)}")
            for seg_id, reason in skipped_segments:
                print(f"    - {seg_id}: {reason}")
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


def cmd_serve(args, config: dict) -> int:
    """Launch the JARVIS web UI (read-only operator console)."""
    from jarvis.web.app import create_app

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 5000)
    debug = getattr(args, "debug", False)

    app = create_app(config)
    print(f"JARVIS web UI running at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JARVIS - Local-first AI assistant for PM workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Set log level to DEBUG for this run",
    )
    parser.add_argument(
        "--retries", type=int, default=1, metavar="N",
        help="Number of retries on LLM parse failure (default: 1, use 0 to fail fast)",
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
        help="(Deprecated no-op) Persistence will be re-added in a future commit.",
    )
    summarize_parser.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing output files before re-running",
    )

    # -- retrieve ---------------------------------------------------------
    retrieve_parser = subparsers.add_parser(
        "retrieve", help="Semantic search over persisted fragment embeddings"
    )
    retrieve_parser.add_argument(
        "--query", "-q", required=True, help="Natural-language search query"
    )
    retrieve_parser.add_argument(
        "--top-k", type=int, default=10, dest="top_k",
        help="Maximum number of results to return (default: 10)",
    )
    retrieve_parser.add_argument(
        "--min-results", type=int, default=3, dest="min_results",
        help="Minimum results to return regardless of score (default: 3)",
    )
    retrieve_parser.add_argument(
        "--min-score", type=float, default=0.50, dest="min_score",
        help="Score threshold — results above this are always included up to --top-k (default: 0.50)",  # noqa: E501
    )

    # -- answer -----------------------------------------------------------
    answer_parser = subparsers.add_parser(
        "answer", help="Answer a question using indexed fragment data"
    )
    answer_parser.add_argument("query", help="Natural-language question")
    answer_parser.add_argument(
        "--top-k", type=int, default=10, dest="top_k",
        help="Maximum number of context excerpts to retrieve (default: 10)",
    )
    answer_parser.add_argument(
        "--min-results", type=int, default=3, dest="min_results",
        help="Minimum excerpts to include regardless of score (default: 3)",
    )
    answer_parser.add_argument(
        "--min-score", type=float, default=0.50, dest="min_score",
        help="Score threshold for including additional excerpts (default: 0.50)",
    )
    answer_parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM sampling temperature (default: 0.3)",
    )
    answer_parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Print context excerpts before the answer",
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
    ingest_chatgpt.add_argument(
        "--persist", action="store_true", default=False,
        help="Persist source file metadata, conversation, and segments to SQLite",
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
        help="Persist segment summaries to SQLite (no Qdrant indexing)",
    )
    ss_chatgpt.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing files and DB records for the range before re-running",
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
        help="Persist topic summaries to SQLite (no Qdrant indexing)",
    )
    dt_chatgpt.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing topic files and DB records before re-running",
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
        help="Persist extracts and statements to SQLite",
    )
    es_chatgpt.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing extract files before re-running",
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
        help="Persist fragments and statement links to SQLite (source of truth). "
             "Does not index Qdrant.",
    )
    fe_chatgpt.add_argument(
        "--embed", action="store_true", default=False,
        help="Embed and index persisted fragments in Qdrant. Requires --persist.",
    )
    fe_chatgpt.add_argument(
        "--force", action="store_true", default=False,
        help="Wipe existing fragment files and records before re-running",
    )

    # -- serve ------------------------------------------------------------
    serve_parser = subparsers.add_parser("serve", help="Launch the JARVIS web UI")
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=5000, help="Bind port (default: 5000)"
    )
    serve_parser.add_argument(
        "--debug", action="store_true", default=False, help="Enable Flask debug mode"
    )

    # -- parse & dispatch -------------------------------------------------
    args = parser.parse_args()

    if (
        getattr(args, "command", None) == "fragment-extracts"
        and getattr(args, "embed", False)
        and not getattr(args, "persist", False)
    ):
        parser.error(
            "--embed requires --persist (Qdrant indexing can only be done for persisted fragments)"
        )

    if not args.command:
        parser.print_help()
        return 1

    config = load_config()
    setup_logging("DEBUG" if args.debug else config["log_level"])

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
    if args.command == "serve":
        return cmd_serve(args, config)

    return 1


if __name__ == "__main__":
    sys.exit(main())
