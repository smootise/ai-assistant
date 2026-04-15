"""CLI entry point for JARVIS."""

import argparse
import logging
import sys
from pathlib import Path

from jarvis.config import load_config
from jarvis.embedder import EmbeddingClient
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

    try:
        ollama_client = OllamaClient(
            model=config["local_model_name"],
            base_url=config["ollama_base_url"],
        )
        output_writer = OutputWriter(
            output_root=config["output_root"],
            timestamp_format=config["output_ts_format"],
            use_timestamp=config["output_timestamp"],
        )
        summarizer = ConversationSummarizer(
            ollama_client=ollama_client,
            output_writer=output_writer,
            prompts_dir=config["prompts_dir"],
            schema=config["schema"],
            schema_version=config["schema_version"],
        )

        output_dir, output_data = summarizer.summarize_file(str(input_file))
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

    return 1


if __name__ == "__main__":
    sys.exit(main())
