"""Output writer for JARVIS summarization artifacts."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)


class OutputWriter:
    """Writes JSON and Markdown summarization outputs."""

    def __init__(
        self,
        output_root: str = "OUTPUTS",
        timestamp_format: str = "%Y%m%d_%H%M%S",
        use_timestamp: bool = True,
    ):
        """Initialize output writer.

        Args:
            output_root: Base output directory.
            timestamp_format: strftime format for timestamped folders.
            use_timestamp: Whether to create timestamped subfolders.
        """
        self.output_root = Path(output_root)
        self.timestamp_format = timestamp_format
        self.use_timestamp = use_timestamp

    def write_outputs(
        self,
        summary_data: Dict[str, Any],
        source_file: str,
        run_id: Optional[str] = None,
    ) -> Path:
        """Write both JSON and Markdown outputs.

        Args:
            summary_data: The summarization result dictionary.
            source_file: Path to the source input file.
            run_id: Optional run identifier.

        Returns:
            Path to the output directory.
        """
        # Create output directory
        if self.use_timestamp:
            timestamp = datetime.now(timezone.utc).strftime(self.timestamp_format)
            output_dir = self.output_root / timestamp
        else:
            output_dir = self.output_root

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing outputs to {output_dir}")

        # Determine base filename from source
        source_path = Path(source_file)
        base_name = source_path.stem  # filename without extension

        # Write JSON artifact
        json_path = output_dir / f"{base_name}.json"
        self._write_json(json_path, summary_data)

        # Write Markdown report
        md_path = output_dir / f"{base_name}.md"
        self._write_markdown(md_path, summary_data)

        logger.info(f"Outputs written successfully:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  Markdown: {md_path}")

        return output_dir

    def _write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Write JSON artifact.

        Args:
            path: Output file path.
            data: Dictionary to serialize.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"JSON written to {path}")

    def _write_markdown(self, path: Path, data: Dict[str, Any]) -> None:
        """Write Markdown report following OUTPUTS.md spec.

        Args:
            path: Output file path.
            data: Summarization data dictionary.
        """
        lines = []

        # 1. Title (from source_file)
        source_file = data.get("source_file", "Unknown")
        title = Path(source_file).stem.replace("_", " ").title()
        lines.append(f"# {title}")
        lines.append("")

        # 2. Confidence (with degraded badge if needed)
        confidence = data.get("confidence", 0.0)
        confidence_line = f"**Confidence:** {confidence:.2f}"
        if data.get("status") == "degraded":
            confidence_line += " *(Degraded)*"
        lines.append(confidence_line)
        lines.append("")

        # 3. Summary section
        lines.append("## Summary")
        lines.append("")
        summary = data.get("summary", "No summary available.")
        lines.append(summary)
        lines.append("")

        # 4. Bullets section
        bullets = data.get("bullets", [])
        if bullets:
            lines.append("## Bullets")
            lines.append("")
            for bullet in bullets:
                lines.append(f"- {bullet}")
            lines.append("")

        # 5. Action Items section
        action_items = data.get("action_items", [])
        lines.append("## Action Items")
        lines.append("")
        if action_items:
            for item in action_items:
                lines.append(f"- [ ] {item}")
        else:
            lines.append("*No action items identified.*")
        lines.append("")

        # 6. Metadata section (2-column table)
        lines.append("## Metadata")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|---|---|")
        lines.append(f"| Provider | {data.get('provider', 'N/A')} |")
        lines.append(f"| Model | {data.get('model', 'N/A')} |")

        latency = data.get("latency_ms")
        latency_str = f"{latency}" if latency is not None else "N/A"
        lines.append(f"| Latency (ms) | {latency_str} |")

        created_at = data.get("created_at", "N/A")
        lines.append(f"| Created At (UTC) | {created_at} |")
        lines.append(f"| Source File | {source_file} |")

        schema = data.get("schema", "N/A")
        schema_version = data.get("schema_version", "N/A")
        lines.append(f"| Schema / Version | {schema} / {schema_version} |")

        lang = data.get("lang")
        if lang:
            lines.append(f"| Language | {lang} |")

        run_id = data.get("run_id")
        if run_id:
            lines.append(f"| Run ID | {run_id} |")

        lines.append("")

        # 7. Warnings section (if present)
        warnings = data.get("warnings", [])
        if warnings:
            lines.append("## Warnings")
            lines.append("")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")

        # Write to file
        content = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Markdown written to {path}")
