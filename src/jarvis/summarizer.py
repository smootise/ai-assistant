"""Conversation summarizer using local Ollama models."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jarvis.ollama import OllamaClient
from jarvis.output_writer import OutputWriter


logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Orchestrates conversation summarization workflow."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        output_writer: OutputWriter,
        prompts_dir: str,
        schema: str,
        schema_version: str,
    ):
        """Initialize summarizer.

        Args:
            ollama_client: Ollama API client.
            output_writer: Output writer for artifacts.
            prompts_dir: Directory containing prompt templates.
            schema: Schema identifier (e.g., 'jarvis.summarization').
            schema_version: Schema version (e.g., '1.0.0').
        """
        self.ollama = ollama_client
        self.output_writer = output_writer
        self.prompts_dir = Path(prompts_dir)
        self.schema = schema
        self.schema_version = schema_version

    def summarize_file(
        self, file_path: str, run_id: str = None, subfolder: Optional[str] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """Summarize a conversation JSON file.

        Args:
            file_path: Path to conversation JSON file.
            run_id: Optional run identifier.

        Returns:
            Tuple of (output_dir, output_data) where output_data is the full
            summarization result dict (matches OUTPUTS.md schema). Callers that
            only need the path can ignore the second element.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            ValueError: If JSON is invalid or required fields are missing.
            ConnectionError: If Ollama is unreachable.
            RuntimeError: If summarization fails.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Starting summarization for {file_path.name}")

        # Track timing
        start_time = time.time()

        # Load conversation
        conversation = self._load_conversation(file_path)

        # Load and format prompt
        system_prompt, user_content = self._build_prompt(conversation)

        # Call Ollama
        raw_response, is_degraded, warning = self.ollama.chat(system_prompt, user_content)

        # Parse JSON response
        try:
            parsed_data, parse_degraded, parse_warning = self.ollama.parse_json_response(
                raw_response
            )
            if parse_degraded:
                is_degraded = True
                warning = parse_warning
        except ValueError as e:
            logger.error(f"Failed to parse model response: {e}")
            raise RuntimeError(f"Model did not return valid JSON: {e}") from e

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Build output document
        output_data = self._build_output_document(
            parsed_data=parsed_data,
            source_file=file_path.name,
            latency_ms=latency_ms,
            is_degraded=is_degraded,
            warning=warning,
            run_id=run_id,
        )

        # Write outputs
        output_dir = self.output_writer.write_outputs(
            summary_data=output_data,
            source_file=file_path.name,
            run_id=run_id,
            subfolder=subfolder,
        )

        logger.info(f"Summarization completed in {latency_ms}ms")
        return output_dir, output_data

    def _load_conversation(self, file_path: Path) -> List[Dict[str, str]]:
        """Load and validate conversation JSON.

        Args:
            file_path: Path to JSON file.

        Returns:
            List of message dictionaries.

        Raises:
            ValueError: If JSON is invalid or malformed.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}") from e

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {file_path}, got {type(data)}")

        if not data:
            logger.warning(f"Empty conversation in {file_path}")

        # Validate message structure
        for i, msg in enumerate(data):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} is not a dict: {type(msg)}")
            if "speaker" not in msg or "content" not in msg:
                raise ValueError(f"Message {i} missing required fields (speaker, content)")

        logger.debug(f"Loaded {len(data)} messages from {file_path}")
        return data

    def _build_prompt(self, conversation: List[Dict[str, str]]) -> tuple:
        """Build system prompt and user content from template and conversation.

        Args:
            conversation: List of message dictionaries.

        Returns:
            Tuple of (system_prompt, user_content).

        Raises:
            FileNotFoundError: If prompt template is missing.
        """
        prompt_path = self.prompts_dir / "summarize_conversation.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        transcript = self._format_transcript(conversation)
        user_content = f"---BEGIN TRANSCRIPT---\n{transcript}\n---END TRANSCRIPT---"

        logger.debug(f"Built system prompt ({len(system_prompt)} chars) + user content ({len(user_content)} chars)")
        return system_prompt, user_content

    def _format_transcript(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation messages as a readable transcript.

        Args:
            conversation: List of message dictionaries.

        Returns:
            Formatted transcript string.
        """
        lines = []
        for msg in conversation:
            speaker = msg.get("speaker", "Unknown")
            content = msg.get("content", "")
            ts = msg.get("ts", "")

            # Format: [Speaker] (timestamp):
            # Content
            if ts:
                lines.append(f"[{speaker}] ({ts}):")
            else:
                lines.append(f"[{speaker}]:")
            lines.append(content)
            lines.append("")  # Blank line between messages

        return "\n".join(lines).strip()

    def _build_output_document(
        self,
        parsed_data: Dict[str, Any],
        source_file: str,
        latency_ms: int,
        is_degraded: bool,
        warning: str,
        run_id: str = None,
    ) -> Dict[str, Any]:
        """Build the complete output document with metadata.

        Args:
            parsed_data: Parsed model output (summary, bullets, etc.).
            source_file: Name of source file.
            latency_ms: Inference latency in milliseconds.
            is_degraded: Whether output is degraded.
            warning: Warning message if degraded.
            run_id: Optional run identifier.

        Returns:
            Complete output dictionary.
        """
        # Start with model output
        output = {
            "summary": parsed_data.get("summary", ""),
            "bullets": parsed_data.get("bullets", []),
            "action_items": parsed_data.get("action_items", []),
            "confidence": parsed_data.get("confidence", 0.0),
        }

        # Add required metadata
        output.update(
            {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "provider": "local",
                "model": self.ollama.model,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source_file": source_file,
            }
        )

        # Add optional metadata
        output["latency_ms"] = latency_ms
        output["source_kind"] = "conversation"

        if is_degraded:
            output["status"] = "degraded"
            output["warnings"] = [warning]
        else:
            output["status"] = "ok"

        if run_id:
            output["run_id"] = run_id

        # Optionally add language hint from parsed data if model included it
        if "lang" in parsed_data:
            output["lang"] = parsed_data["lang"]

        return output
