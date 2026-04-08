"""CLI entry point for JARVIS."""

import argparse
import logging
import sys
from pathlib import Path

from jarvis.config import load_config
from jarvis.ollama import OllamaClient
from jarvis.output_writer import OutputWriter
from jarvis.summarizer import ConversationSummarizer


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


def cmd_summarize(args: argparse.Namespace, config: dict) -> int:
    """Execute summarize command.

    Args:
        args: Parsed command-line arguments.
        config: Loaded configuration dictionary.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger = logging.getLogger(__name__)

    # Validate input file
    input_file = Path(args.file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    if input_file.suffix.lower() != ".json":
        logger.error(f"Unsupported file type: {input_file.suffix} (expected .json)")
        return 1

    # Initialize components
    try:
        ollama_client = OllamaClient(model=config["local_model_name"])

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

        # Run summarization
        output_dir = summarizer.summarize_file(str(input_file))
        logger.info(f"Success! Outputs saved to: {output_dir}")
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

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a conversation or note"
    )
    summarize_parser.add_argument(
        "--file", "-f", required=True, help="Path to input file (JSON conversation)"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config["log_level"])

    # Route to command handler
    if args.command == "summarize":
        return cmd_summarize(args, config)

    return 1


if __name__ == "__main__":
    sys.exit(main())
