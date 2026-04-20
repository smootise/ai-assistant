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
_SCHEMA_VERSION = 3

_DDL = """
CREATE TABLE IF NOT EXISTS _jarvis_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT,
    source_file             TEXT    NOT NULL,
    source_kind             TEXT    NOT NULL,
    provider                TEXT    NOT NULL,
    model                   TEXT    NOT NULL,
    embedding_model         TEXT,
    schema                  TEXT    NOT NULL,
    schema_version          TEXT    NOT NULL,
    status                  TEXT    NOT NULL,
    lang                    TEXT,
    confidence              REAL    NOT NULL,
    latency_ms              INTEGER,
    summary                 TEXT    NOT NULL,
    bullets                 TEXT    NOT NULL,
    action_items            TEXT    NOT NULL,
    warnings                TEXT,
    created_at              TEXT    NOT NULL,
    embedded_at             TEXT,
    output_json_path        TEXT,
    output_md_path          TEXT,
    qdrant_point_id         TEXT,
    chunk_id                TEXT,
    chunk_index             INTEGER,
    parent_conversation_id  TEXT,
    segment_index           INTEGER,
    segment_chunk_range     TEXT
);

CREATE INDEX IF NOT EXISTS idx_summaries_source_file  ON summaries (source_file);
CREATE INDEX IF NOT EXISTS idx_summaries_created_at   ON summaries (created_at);
CREATE INDEX IF NOT EXISTS idx_summaries_status       ON summaries (status);
"""

# Migration DDL (each list is idempotent via try/except on ALTER TABLE).
# Indexes on new columns are added here, not in _DDL, so they only run after
# the columns exist on old databases.
_MIGRATION_V2 = [
    "ALTER TABLE summaries ADD COLUMN chunk_id               TEXT",
    "ALTER TABLE summaries ADD COLUMN chunk_index            INTEGER",
    "ALTER TABLE summaries ADD COLUMN parent_conversation_id TEXT",
    "CREATE INDEX IF NOT EXISTS idx_summaries_chunk_id    ON summaries (chunk_id)",
    "CREATE INDEX IF NOT EXISTS idx_summaries_parent_conv ON summaries (parent_conversation_id)",
]

_MIGRATION_V3 = [
    "ALTER TABLE summaries ADD COLUMN segment_index       INTEGER",
    "ALTER TABLE summaries ADD COLUMN segment_chunk_range TEXT",
    "CREATE INDEX IF NOT EXISTS idx_summaries_segment_index ON summaries (segment_index)",
]


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
            "chunk_id": output_data.get("chunk_id"),
            "chunk_index": output_data.get("chunk_index"),
            "parent_conversation_id": output_data.get("parent_conversation_id"),
            "segment_index": output_data.get("segment_index"),
            "segment_chunk_range": output_data.get("segment_chunk_range"),
        }

        sql = """
            INSERT INTO summaries (
                run_id, source_file, source_kind, provider, model, embedding_model,
                schema, schema_version, status, lang, confidence, latency_ms,
                summary, bullets, action_items, warnings,
                created_at, embedded_at, output_json_path, output_md_path, qdrant_point_id,
                chunk_id, chunk_index, parent_conversation_id,
                segment_index, segment_chunk_range
            ) VALUES (
                :run_id, :source_file, :source_kind, :provider, :model, :embedding_model,
                :schema, :schema_version, :status, :lang, :confidence, :latency_ms,
                :summary, :bullets, :action_items, :warnings,
                :created_at, :embedded_at, :output_json_path, :output_md_path, :qdrant_point_id,
                :chunk_id, :chunk_index, :parent_conversation_id,
                :segment_index, :segment_chunk_range
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

    def get_chunk_summaries_by_conversation(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch all chunk summaries for a conversation, ordered by chunk_index.

        Args:
            conversation_id: The parent conversation ID.

        Returns:
            List of row dicts ordered by chunk_index ASC.
        """
        sql = """
            SELECT * FROM summaries
            WHERE parent_conversation_id = ? AND chunk_id IS NOT NULL
            ORDER BY chunk_index ASC
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (conversation_id,)).fetchall()

        result = []
        for row in rows:
            r = dict(row)
            r["bullets"] = json.loads(r["bullets"] or "[]")
            r["action_items"] = json.loads(r["action_items"] or "[]")
            r["warnings"] = json.loads(r["warnings"]) if r["warnings"] else []
            result.append(r)
        return result

    def get_segment_summaries_by_conversation(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch all segment summaries for a conversation, ordered by segment_index.

        Args:
            conversation_id: The parent conversation ID.

        Returns:
            List of row dicts ordered by segment_index ASC.
        """
        sql = """
            SELECT * FROM summaries
            WHERE parent_conversation_id = ? AND source_kind = 'ai_chat_segment'
            ORDER BY segment_index ASC
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (conversation_id,)).fetchall()

        result = []
        for row in rows:
            r = dict(row)
            r["bullets"] = json.loads(r["bullets"] or "[]")
            r["action_items"] = json.loads(r["action_items"] or "[]")
            r["warnings"] = json.loads(r["warnings"]) if r["warnings"] else []
            result.append(r)
        return result

    def get_segment_rows(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Fetch all segment summary rows for a conversation.

        Args:
            conversation_id: The parent conversation ID.

        Returns:
            List of row dicts.
        """
        sql = """
            SELECT * FROM summaries
            WHERE parent_conversation_id = ? AND source_kind = 'ai_chat_segment'
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (conversation_id,)).fetchall()
        return [dict(row) for row in rows]

    def delete_segment_rows(self, conversation_id: str) -> List[str]:
        """Delete all segment summary rows and return their qdrant_point_ids.

        Args:
            conversation_id: The parent conversation ID.

        Returns:
            List of qdrant_point_id strings for the deleted rows.
        """
        rows = self.get_segment_rows(conversation_id)
        point_ids = [r["qdrant_point_id"] for r in rows if r.get("qdrant_point_id")]
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM summaries
                WHERE parent_conversation_id = ? AND source_kind = 'ai_chat_segment'
                """,
                (conversation_id,),
            )
        logger.info(f"Deleted {len(rows)} segment rows for conversation {conversation_id}")
        return point_ids

    def get_by_source_file(self, source_file: str) -> Optional[Dict[str, Any]]:
        """Fetch a single summary row by source_file.

        Args:
            source_file: The source_file value to look up.

        Returns:
            Row dict or None if not found.
        """
        sql = "SELECT * FROM summaries WHERE source_file = ? LIMIT 1"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (source_file,)).fetchone()
        if row is None:
            return None
        r = dict(row)
        r["bullets"] = json.loads(r["bullets"] or "[]")
        r["action_items"] = json.loads(r["action_items"] or "[]")
        r["warnings"] = json.loads(r["warnings"]) if r["warnings"] else []
        return r

    def delete_by_source_file(self, source_file: str) -> Optional[str]:
        """Delete a summary row by source_file and return its qdrant_point_id.

        Args:
            source_file: The source_file value to delete.

        Returns:
            qdrant_point_id if the row existed, else None.
        """
        row = self.get_by_source_file(source_file)
        if row is None:
            return None
        qdrant_point_id = row.get("qdrant_point_id")
        with self._connect() as conn:
            conn.execute("DELETE FROM summaries WHERE source_file = ?", (source_file,))
        logger.info(f"Deleted summary row for source_file={source_file}")
        return qdrant_point_id

    def get_chunk_rows_by_ids(
        self, conversation_id: str, chunk_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch chunk summary rows for specific chunk_ids within a conversation.

        Args:
            conversation_id: The parent conversation ID.
            chunk_ids: List of chunk_id values to fetch.

        Returns:
            List of row dicts.
        """
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        sql = f"""
            SELECT * FROM summaries
            WHERE parent_conversation_id = ? AND chunk_id IN ({placeholders})
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, [conversation_id] + chunk_ids).fetchall()
        result = []
        for row in rows:
            r = dict(row)
            r["bullets"] = json.loads(r["bullets"] or "[]")
            r["action_items"] = json.loads(r["action_items"] or "[]")
            r["warnings"] = json.loads(r["warnings"]) if r["warnings"] else []
            result.append(r)
        return result

    def delete_chunk_rows(
        self, conversation_id: str, chunk_ids: List[str]
    ) -> List[str]:
        """Delete chunk summary rows and return their qdrant_point_ids.

        Args:
            conversation_id: The parent conversation ID.
            chunk_ids: List of chunk_id values to delete.

        Returns:
            List of qdrant_point_id strings for the deleted rows (may include None values
            filtered out).
        """
        rows = self.get_chunk_rows_by_ids(conversation_id, chunk_ids)
        point_ids = [r["qdrant_point_id"] for r in rows if r.get("qdrant_point_id")]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            sql = f"""
                DELETE FROM summaries
                WHERE parent_conversation_id = ? AND chunk_id IN ({placeholders})
            """
            with self._connect() as conn:
                conn.execute(sql, [conversation_id] + chunk_ids)
        logger.info(
            f"Deleted {len(rows)} chunk rows for conversation {conversation_id}"
        )
        return point_ids

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
        """Create tables and indexes if they don't exist, then run migrations."""
        with self._connect() as conn:
            conn.executescript(_DDL)
            self._migrate_db(conn)
        logger.debug(f"SQLite store ready at {self.db_path}")

    def _migrate_db(self, conn: sqlite3.Connection) -> None:
        """Apply incremental schema migrations to existing databases.

        Reads the stored schema version from _jarvis_meta and runs any
        migrations needed to reach _SCHEMA_VERSION.

        Args:
            conn: Open SQLite connection (within an active transaction context).
        """
        row = conn.execute(
            "SELECT value FROM _jarvis_meta WHERE key = 'schema_version'"
        ).fetchone()
        stored_version = int(row[0]) if row else 1

        if stored_version < 2:
            for ddl in _MIGRATION_V2:
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # Column already exists — idempotent

        if stored_version < 3:
            for ddl in _MIGRATION_V3:
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # Column already exists — idempotent

        if stored_version < _SCHEMA_VERSION:
            conn.execute(
                "INSERT OR REPLACE INTO _jarvis_meta (key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            logger.info(f"Migrated SQLite schema from v{stored_version} to v{_SCHEMA_VERSION}")
