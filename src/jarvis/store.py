"""SQLite persistence layer for JARVIS summaries.

SQLite is the source of truth for all structured summary records.
Each row maps 1:1 to one summarization run.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)

# Current schema version — increment when adding columns.
_SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS summaries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    source_file       TEXT    NOT NULL,
    source_kind       TEXT    NOT NULL,
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    embedding_model   TEXT,
    schema            TEXT    NOT NULL,
    schema_version    TEXT    NOT NULL,
    status            TEXT    NOT NULL,
    lang              TEXT,
    confidence        REAL    NOT NULL,
    latency_ms        INTEGER,
    summary           TEXT    NOT NULL,
    bullets           TEXT    NOT NULL,
    action_items      TEXT    NOT NULL,
    warnings          TEXT,
    created_at        TEXT    NOT NULL,
    embedded_at       TEXT,
    output_json_path  TEXT,
    output_md_path    TEXT,
    qdrant_point_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_summaries_source_file ON summaries (source_file);
CREATE INDEX IF NOT EXISTS idx_summaries_created_at  ON summaries (created_at);
CREATE INDEX IF NOT EXISTS idx_summaries_status      ON summaries (status);
"""


class SummaryStore:
    """Manages the SQLite summaries table."""

    def __init__(self, db_path: str):
        """Initialize and migrate the database.

        Args:
            db_path: Path to the SQLite database file. Parent dirs are created
                     automatically.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_summary(
        self,
        output_data: Dict[str, Any],
        output_json_path: Optional[str] = None,
        output_md_path: Optional[str] = None,
    ) -> int:
        """Insert a summary record and return its row ID.

        Args:
            output_data: The summarization output dict (matches OUTPUTS.md schema).
            output_json_path: Repo-relative path to the .json artifact.
            output_md_path: Repo-relative path to the .md artifact.

        Returns:
            Auto-incremented row ID of the inserted record.
        """
        row = {
            "run_id": output_data.get("run_id"),
            "source_file": output_data["source_file"],
            "source_kind": output_data.get("source_kind", "conversation"),
            "provider": output_data["provider"],
            "model": output_data["model"],
            "embedding_model": output_data.get("embedding_model"),
            "schema": output_data.get("schema", "jarvis.summarization"),
            "schema_version": output_data.get("schema_version", "1.0.0"),
            "status": output_data.get("status", "ok"),
            "lang": output_data.get("lang"),
            "confidence": output_data["confidence"],
            "latency_ms": output_data.get("latency_ms"),
            "summary": output_data["summary"],
            "bullets": json.dumps(output_data.get("bullets", []), ensure_ascii=False),
            "action_items": json.dumps(
                output_data.get("action_items", []), ensure_ascii=False
            ),
            "warnings": json.dumps(output_data.get("warnings", []), ensure_ascii=False)
            if output_data.get("warnings")
            else None,
            "created_at": output_data["created_at"],
            "embedded_at": None,
            "output_json_path": output_json_path,
            "output_md_path": output_md_path,
            "qdrant_point_id": None,
        }

        sql = """
            INSERT INTO summaries (
                run_id, source_file, source_kind, provider, model, embedding_model,
                schema, schema_version, status, lang, confidence, latency_ms,
                summary, bullets, action_items, warnings,
                created_at, embedded_at, output_json_path, output_md_path, qdrant_point_id
            ) VALUES (
                :run_id, :source_file, :source_kind, :provider, :model, :embedding_model,
                :schema, :schema_version, :status, :lang, :confidence, :latency_ms,
                :summary, :bullets, :action_items, :warnings,
                :created_at, :embedded_at, :output_json_path, :output_md_path, :qdrant_point_id
            )
        """
        with self._connect() as conn:
            cursor = conn.execute(sql, row)
            row_id = cursor.lastrowid

        logger.info(f"Inserted summary record id={row_id} for {output_data['source_file']}")
        return row_id

    def update_embedding(
        self,
        summary_id: int,
        qdrant_point_id: str,
        embedding_model: str,
    ) -> None:
        """Update a row with Qdrant point ID and embedding metadata.

        Args:
            summary_id: Row ID to update.
            qdrant_point_id: UUID of the Qdrant point.
            embedding_model: Name of the embedding model used.
        """
        embedded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        sql = """
            UPDATE summaries
            SET qdrant_point_id = ?, embedding_model = ?, embedded_at = ?
            WHERE id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, (qdrant_point_id, embedding_model, embedded_at, summary_id))

        logger.debug(
            f"Updated summary id={summary_id} with "
            f"qdrant_point_id={qdrant_point_id}, embedded_at={embedded_at}"
        )

    def get_by_ids(self, summary_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch summary rows by a list of IDs, preserving order.

        Args:
            summary_ids: List of row IDs to fetch.

        Returns:
            List of row dicts in the same order as summary_ids.
        """
        if not summary_ids:
            return []

        placeholders = ",".join("?" * len(summary_ids))
        sql = f"SELECT * FROM summaries WHERE id IN ({placeholders})"

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, summary_ids).fetchall()

        # Re-order to match the requested order (important for ranked retrieval)
        id_to_row = {row["id"]: dict(row) for row in rows}
        ordered = [id_to_row[sid] for sid in summary_ids if sid in id_to_row]

        # Deserialize JSON fields
        for row in ordered:
            row["bullets"] = json.loads(row["bullets"] or "[]")
            row["action_items"] = json.loads(row["action_items"] or "[]")
            row["warnings"] = json.loads(row["warnings"]) if row["warnings"] else []

        return ordered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode for better concurrency."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._connect() as conn:
            conn.executescript(_DDL)
        logger.debug(f"SQLite store ready at {self.db_path}")
