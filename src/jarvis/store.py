"""SQLite persistence layer for JARVIS.

SQLite is the relational source of truth for all pipeline records.
One table per entity; foreign keys enforced. Full records are always
reconstructible from SQLite by ID.

Table hierarchy:
  source_files
    └── conversations
          └── segments
                ├── segment_summaries
                ├── extracts
                │     ├── extract_statements
                │     └── fragments
                │           └── fragment_statement_links ──► extract_statements
                └── topic_summaries
                      └── topic_segments ──► segments
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 8

_DDL = """
CREATE TABLE IF NOT EXISTS _jarvis_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- File metadata: raw export + normalized.json (content stays on disk)
CREATE TABLE IF NOT EXISTS source_files (
    source_file_id  TEXT PRIMARY KEY,
    source_kind     TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    storage_path    TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER,
    ingested_at     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE(sha256, source_kind)
);

-- One row per conversation
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id            TEXT PRIMARY KEY,
    raw_source_file_id         TEXT REFERENCES source_files(source_file_id),
    normalized_source_file_id  TEXT REFERENCES source_files(source_file_id),
    title                      TEXT,
    conversation_date          TEXT,
    source_platform            TEXT NOT NULL,
    message_count              INTEGER,
    imported_at                TEXT,
    created_at                 TEXT NOT NULL
);

-- One row per segment (includes full text for citation rendering)
CREATE TABLE IF NOT EXISTS segments (
    segment_id       TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(conversation_id),
    segment_index    INTEGER NOT NULL,
    start_position   INTEGER NOT NULL,
    end_position     INTEGER NOT NULL,
    message_ids_json TEXT NOT NULL,
    conversation_date TEXT,
    segment_text     TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    UNIQUE(conversation_id, segment_index)
);

-- One segment-summary row per segment (SQLite only, never embedded)
CREATE TABLE IF NOT EXISTS segment_summaries (
    segment_summary_id TEXT PRIMARY KEY,
    segment_id         TEXT NOT NULL REFERENCES segments(segment_id),
    summary            TEXT NOT NULL,
    bullets            TEXT NOT NULL DEFAULT '[]',
    action_items       TEXT NOT NULL DEFAULT '[]',
    lang               TEXT,
    status             TEXT NOT NULL,
    warnings           TEXT,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    latency_ms         INTEGER,
    created_at         TEXT NOT NULL,
    UNIQUE(segment_id)
);

-- One extract row per segment
CREATE TABLE IF NOT EXISTS extracts (
    extract_id             TEXT PRIMARY KEY,
    segment_id             TEXT NOT NULL REFERENCES segments(segment_id),
    segment_index          INTEGER NOT NULL,
    parent_conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    provider               TEXT NOT NULL,
    model                  TEXT NOT NULL,
    prompt_version         TEXT,
    status                 TEXT NOT NULL,
    warnings               TEXT,
    latency_ms             INTEGER,
    created_at             TEXT NOT NULL,
    UNIQUE(segment_id)
);

-- One row per extracted statement (ordered by statement_index)
CREATE TABLE IF NOT EXISTS extract_statements (
    statement_id    TEXT PRIMARY KEY,
    extract_id      TEXT NOT NULL REFERENCES extracts(extract_id) ON DELETE CASCADE,
    statement_index INTEGER NOT NULL,
    speaker         TEXT NOT NULL,
    text            TEXT NOT NULL,
    UNIQUE(extract_id, statement_index)
);

-- One row per fragment
CREATE TABLE IF NOT EXISTS fragments (
    fragment_id     TEXT PRIMARY KEY,
    extract_id      TEXT NOT NULL REFERENCES extracts(extract_id),
    fragment_index  INTEGER NOT NULL,
    title           TEXT,
    retrieval_text  TEXT NOT NULL,
    status          TEXT NOT NULL,
    qdrant_point_id TEXT,
    embedded_at     TEXT,
    embedding_model TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(extract_id, fragment_index)
);

-- Many-to-many: fragment membership and ordering (supports non-contiguous spans later)
CREATE TABLE IF NOT EXISTS fragment_statement_links (
    fragment_id          TEXT NOT NULL REFERENCES fragments(fragment_id) ON DELETE CASCADE,
    statement_id         TEXT NOT NULL REFERENCES extract_statements(statement_id),
    position_in_fragment INTEGER NOT NULL,
    PRIMARY KEY (fragment_id, statement_id),
    UNIQUE (fragment_id, position_in_fragment)
);

-- Topic summaries (SQLite only, never embedded)
CREATE TABLE IF NOT EXISTS topic_summaries (
    topic_id            TEXT PRIMARY KEY,
    conversation_id     TEXT NOT NULL REFERENCES conversations(conversation_id),
    topic_index         INTEGER NOT NULL,
    topic_segment_range TEXT,
    summary             TEXT NOT NULL,
    bullets             TEXT NOT NULL DEFAULT '[]',
    action_items        TEXT NOT NULL DEFAULT '[]',
    lang                TEXT,
    status              TEXT NOT NULL,
    warnings            TEXT,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    latency_ms          INTEGER,
    created_at          TEXT NOT NULL,
    UNIQUE(conversation_id, topic_index)
);

-- Mapping: which segments belong to which topic
CREATE TABLE IF NOT EXISTS topic_segments (
    topic_id   TEXT NOT NULL REFERENCES topic_summaries(topic_id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL REFERENCES segments(segment_id),
    position   INTEGER NOT NULL,
    PRIMARY KEY (topic_id, segment_id)
);

CREATE INDEX IF NOT EXISTS idx_segments_conv       ON segments (conversation_id, segment_index);
CREATE INDEX IF NOT EXISTS idx_extracts_conv       ON extracts (parent_conversation_id);
CREATE INDEX IF NOT EXISTS idx_fragments_extract   ON fragments (extract_id);
CREATE INDEX IF NOT EXISTS idx_links_statement     ON fragment_statement_links (statement_id);
CREATE INDEX IF NOT EXISTS idx_fragments_qdrant    ON fragments (qdrant_point_id);
CREATE INDEX IF NOT EXISTS idx_seg_sum_segment     ON segment_summaries (segment_id);
CREATE INDEX IF NOT EXISTS idx_topic_sum_conv
    ON topic_summaries (conversation_id, topic_index);
CREATE INDEX IF NOT EXISTS idx_topic_segs_topic    ON topic_segments (topic_id);

-- Upload + ingest job tracking
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    job_type        TEXT NOT NULL,
    status          TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
    input_metadata  TEXT NOT NULL,
    result          TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at DESC);
"""

# No migration path — DB is always wiped and rebuilt on schema changes.


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SummaryStore:
    """Relational SQLite store for all JARVIS pipeline records."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Source files
    # ------------------------------------------------------------------

    def insert_source_file(self, data: Dict[str, Any]) -> str:
        """Insert a source file metadata row. Idempotent on (sha256, source_kind).

        Returns the source_file_id (sha256).
        """
        sql = """
            INSERT OR IGNORE INTO source_files (
                source_file_id, source_kind, original_filename, storage_path,
                sha256, size_bytes, ingested_at, created_at
            ) VALUES (
                :source_file_id, :source_kind, :original_filename, :storage_path,
                :sha256, :size_bytes, :ingested_at, :created_at
            )
        """
        now = _now_iso()
        row = {
            "source_file_id": data["source_file_id"],
            "source_kind": data["source_kind"],
            "original_filename": data["original_filename"],
            "storage_path": data["storage_path"],
            "sha256": data["sha256"],
            "size_bytes": data.get("size_bytes"),
            "ingested_at": data.get("ingested_at") or now,
            "created_at": data.get("created_at") or now,
        }
        with self._connect() as conn:
            conn.execute(sql, row)
        return data["source_file_id"]

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def insert_conversation(self, data: Dict[str, Any]) -> None:
        """Insert a conversation row. Idempotent on conversation_id."""
        sql = """
            INSERT OR IGNORE INTO conversations (
                conversation_id, raw_source_file_id, normalized_source_file_id,
                title, conversation_date, source_platform, message_count,
                imported_at, created_at
            ) VALUES (
                :conversation_id, :raw_source_file_id, :normalized_source_file_id,
                :title, :conversation_date, :source_platform, :message_count,
                :imported_at, :created_at
            )
        """
        now = _now_iso()
        row = {
            "conversation_id": data["conversation_id"],
            "raw_source_file_id": data.get("raw_source_file_id"),
            "normalized_source_file_id": data.get("normalized_source_file_id"),
            "title": data.get("title"),
            "conversation_date": data.get("conversation_date"),
            "source_platform": data.get("source_platform") or "chatgpt",
            "message_count": data.get("message_count"),
            "imported_at": data.get("imported_at") or now,
            "created_at": data.get("created_at") or now,
        }
        with self._connect() as conn:
            conn.execute(sql, row)

    # ------------------------------------------------------------------
    # Segments
    # ------------------------------------------------------------------

    def insert_segment(self, data: Dict[str, Any]) -> str:
        """Insert a segment row. Idempotent on segment_id.

        Returns the segment_id.
        """
        sql = """
            INSERT OR IGNORE INTO segments (
                segment_id, conversation_id, segment_index,
                start_position, end_position, message_ids_json,
                conversation_date, segment_text, created_at
            ) VALUES (
                :segment_id, :conversation_id, :segment_index,
                :start_position, :end_position, :message_ids_json,
                :conversation_date, :segment_text, :created_at
            )
        """
        row = {
            "segment_id": data["segment_id"],
            "conversation_id": data["conversation_id"],
            "segment_index": data["segment_index"],
            "start_position": data["start_position"],
            "end_position": data["end_position"],
            "message_ids_json": json.dumps(data.get("message_ids", []), ensure_ascii=False),
            "conversation_date": data.get("conversation_date"),
            "segment_text": data["segment_text"],
            "created_at": data.get("created_at") or _now_iso(),
        }
        with self._connect() as conn:
            conn.execute(sql, row)
        return data["segment_id"]

    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single segment row by ID."""
        sql = "SELECT * FROM segments WHERE segment_id = ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (segment_id,)).fetchone()
        if row is None:
            return None
        r = dict(row)
        r["message_ids"] = json.loads(r.get("message_ids_json") or "[]")
        return r

    def delete_segments(self, conversation_id: str, segment_ids: List[str]) -> None:
        """Delete segment rows by ID."""
        if not segment_ids:
            return
        placeholders = ",".join("?" * len(segment_ids))
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM segments WHERE conversation_id = ? "
                f"AND segment_id IN ({placeholders})",
                [conversation_id] + segment_ids,
            )

    # ------------------------------------------------------------------
    # Segment summaries
    # ------------------------------------------------------------------

    def insert_segment_summary(self, data: Dict[str, Any]) -> str:
        """Insert a segment summary row. Idempotent on segment_id.

        Returns the segment_summary_id.
        """
        sql = """
            INSERT OR IGNORE INTO segment_summaries (
                segment_summary_id, segment_id, summary, bullets, action_items,
                lang, status, warnings, provider, model, latency_ms, created_at
            ) VALUES (
                :segment_summary_id, :segment_id, :summary, :bullets, :action_items,
                :lang, :status, :warnings, :provider, :model, :latency_ms, :created_at
            )
        """
        segment_id = data["segment_id"]
        row = {
            "segment_summary_id": f"{segment_id}_ss",
            "segment_id": segment_id,
            "summary": data.get("summary") or "",
            "bullets": json.dumps(data.get("bullets") or [], ensure_ascii=False),
            "action_items": json.dumps(data.get("action_items") or [], ensure_ascii=False),
            "lang": data.get("lang"),
            "status": data.get("status") or "ok",
            "warnings": json.dumps(data.get("warnings", []), ensure_ascii=False)
            if data.get("warnings") else None,
            "provider": data.get("provider") or "local",
            "model": data.get("model") or "",
            "latency_ms": data.get("latency_ms"),
            "created_at": data.get("created_at") or _now_iso(),
        }
        with self._connect() as conn:
            conn.execute(sql, row)
        return row["segment_summary_id"]

    def get_segment_summary(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a segment summary by segment_id."""
        sql = "SELECT * FROM segment_summaries WHERE segment_id = ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (segment_id,)).fetchone()
        if row is None:
            return None
        r = dict(row)
        r["bullets"] = json.loads(r["bullets"] or "[]")
        r["action_items"] = json.loads(r["action_items"] or "[]")
        r["warnings"] = json.loads(r["warnings"]) if r["warnings"] else []
        return r

    def delete_segment_summaries(self, conversation_id: str) -> None:
        """Delete all segment summaries for a conversation."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM segment_summaries WHERE segment_id IN "
                "(SELECT segment_id FROM segments WHERE conversation_id = ?)",
                (conversation_id,),
            )

    # ------------------------------------------------------------------
    # Extracts
    # ------------------------------------------------------------------

    def insert_extract(self, data: Dict[str, Any]) -> str:
        """Insert an extract row. Idempotent on segment_id (UNIQUE constraint).

        Returns the extract_id.
        """
        sql = """
            INSERT OR IGNORE INTO extracts (
                extract_id, segment_id, segment_index, parent_conversation_id,
                provider, model, prompt_version, status, warnings, latency_ms, created_at
            ) VALUES (
                :extract_id, :segment_id, :segment_index, :parent_conversation_id,
                :provider, :model, :prompt_version, :status, :warnings, :latency_ms, :created_at
            )
        """
        segment_id = data["segment_id"]
        extract_id = f"{segment_id}_x"
        row = {
            "extract_id": extract_id,
            "segment_id": segment_id,
            "segment_index": data.get("segment_index") or 0,
            "parent_conversation_id": data["parent_conversation_id"],
            "provider": data.get("provider") or "local",
            "model": data.get("model") or "",
            "prompt_version": data.get("prompt_version"),
            "status": data.get("status") or "ok",
            "warnings": json.dumps(data.get("warnings", []), ensure_ascii=False)
            if data.get("warnings") else None,
            "latency_ms": data.get("latency_ms"),
            "created_at": data.get("created_at") or _now_iso(),
        }
        with self._connect() as conn:
            conn.execute(sql, row)
        return extract_id

    def get_extract_by_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the extract row for a segment."""
        sql = "SELECT * FROM extracts WHERE segment_id = ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (segment_id,)).fetchone()
        if row is None:
            return None
        r = dict(row)
        r["warnings"] = json.loads(r["warnings"]) if r["warnings"] else []
        return r

    def delete_extracts(
        self,
        conversation_id: str,
        segment_indices: Optional[List[int]] = None,
    ) -> List[str]:
        """Delete extract rows for a conversation, returning Qdrant point IDs of any
        fragments that were removed so callers can clean the vector index.

        Deletes fragments (+ fragment_statement_links via cascade) first to satisfy the
        FK constraint on fragments.extract_id, then deletes the extracts (which cascades
        to extract_statements).
        """
        with self._connect() as conn:
            if segment_indices is not None:
                placeholders = ",".join("?" * len(segment_indices))
                extract_id_rows = conn.execute(
                    f"SELECT extract_id FROM extracts WHERE parent_conversation_id = ? "
                    f"AND segment_index IN ({placeholders})",
                    [conversation_id] + segment_indices,
                ).fetchall()
            else:
                extract_id_rows = conn.execute(
                    "SELECT extract_id FROM extracts WHERE parent_conversation_id = ?",
                    (conversation_id,),
                ).fetchall()

            extract_ids = [r[0] for r in extract_id_rows]
            point_ids: List[str] = []

            if extract_ids:
                ext_placeholders = ",".join("?" * len(extract_ids))
                point_rows = conn.execute(
                    f"SELECT qdrant_point_id FROM fragments "
                    f"WHERE extract_id IN ({ext_placeholders}) "
                    f"AND qdrant_point_id IS NOT NULL",
                    extract_ids,
                ).fetchall()
                point_ids = [r[0] for r in point_rows]

                conn.execute(
                    f"DELETE FROM fragments WHERE extract_id IN ({ext_placeholders})",
                    extract_ids,
                )
                conn.execute(
                    f"DELETE FROM extracts WHERE extract_id IN ({ext_placeholders})",
                    extract_ids,
                )

        return point_ids

    # ------------------------------------------------------------------
    # Extract statements
    # ------------------------------------------------------------------

    def insert_statements(self, extract_id: str, statements: List[Dict[str, Any]]) -> None:
        """Insert extract statement rows. Idempotent on (extract_id, statement_index)."""
        sql = """
            INSERT OR IGNORE INTO extract_statements (
                statement_id, extract_id, statement_index, speaker, text
            ) VALUES (:statement_id, :extract_id, :statement_index, :speaker, :text)
        """
        rows = []
        for s in statements:
            idx = s["statement_index"]
            rows.append({
                "statement_id": f"{extract_id}_st{idx:04d}",
                "extract_id": extract_id,
                "statement_index": idx,
                "speaker": s["speaker"],
                "text": s["text"],
            })
        if rows:
            with self._connect() as conn:
                conn.executemany(sql, rows)

    def get_statements_for_extract(self, extract_id: str) -> List[Dict[str, Any]]:
        """Return statements for an extract, ordered by statement_index."""
        sql = """
            SELECT * FROM extract_statements
            WHERE extract_id = ?
            ORDER BY statement_index ASC
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (extract_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Fragments
    # ------------------------------------------------------------------

    def insert_fragment(self, data: Dict[str, Any]) -> str:
        """Insert a fragment row. Idempotent on (extract_id, fragment_index).

        Returns the fragment_id.
        """
        sql = """
            INSERT OR IGNORE INTO fragments (
                fragment_id, extract_id, fragment_index, title, retrieval_text,
                status, qdrant_point_id, embedded_at, embedding_model, created_at
            ) VALUES (
                :fragment_id, :extract_id, :fragment_index, :title, :retrieval_text,
                :status, :qdrant_point_id, :embedded_at, :embedding_model, :created_at
            )
        """
        extract_id = data["extract_id"]
        frag_idx = data["fragment_index"]
        fragment_id = f"{extract_id}_f{frag_idx:03d}"
        row = {
            "fragment_id": fragment_id,
            "extract_id": extract_id,
            "fragment_index": frag_idx,
            "title": data.get("title"),
            "retrieval_text": data["retrieval_text"],
            "status": data.get("status") or "ok",
            "qdrant_point_id": data.get("qdrant_point_id"),
            "embedded_at": data.get("embedded_at"),
            "embedding_model": data.get("embedding_model"),
            "created_at": data.get("created_at") or _now_iso(),
        }
        with self._connect() as conn:
            conn.execute(sql, row)
        return fragment_id

    def get_fragment(self, fragment_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single fragment row."""
        sql = "SELECT * FROM fragments WHERE fragment_id = ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (fragment_id,)).fetchone()
        return dict(row) if row else None

    def update_fragment_embedding(
        self,
        fragment_id: str,
        qdrant_point_id: str,
        embedding_model: str,
    ) -> None:
        """Write Qdrant point ID and embedding metadata back to the fragment row."""
        embedded_at = _now_iso()
        sql = """
            UPDATE fragments
            SET qdrant_point_id = ?, embedding_model = ?, embedded_at = ?
            WHERE fragment_id = ?
        """
        with self._connect() as conn:
            conn.execute(sql, (qdrant_point_id, embedding_model, embedded_at, fragment_id))
        logger.debug(f"Fragment {fragment_id} updated: qdrant_point_id={qdrant_point_id}")

    def get_fragments_for_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Fetch all fragment rows for a conversation (for --force cleanup)."""
        sql = """
            SELECT f.*
            FROM fragments f
            JOIN extracts e ON f.extract_id = e.extract_id
            WHERE e.parent_conversation_id = ?
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (conversation_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_fragments(self, conversation_id: str) -> List[str]:
        """Delete all fragment rows for a conversation; return their qdrant_point_ids."""
        rows = self.get_fragments_for_conversation(conversation_id)
        point_ids = [r["qdrant_point_id"] for r in rows if r.get("qdrant_point_id")]
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM fragments WHERE extract_id IN "
                "(SELECT extract_id FROM extracts WHERE parent_conversation_id = ?)",
                (conversation_id,),
            )
        logger.info(f"Deleted {len(rows)} fragment rows for conversation {conversation_id}")
        return point_ids

    # ------------------------------------------------------------------
    # Fragment statement links
    # ------------------------------------------------------------------

    def get_link_count(self, fragment_id: str) -> int:
        """Return the number of statement links for a fragment."""
        sql = "SELECT COUNT(*) FROM fragment_statement_links WHERE fragment_id = ?"
        with self._connect() as conn:
            return conn.execute(sql, (fragment_id,)).fetchone()[0]

    def insert_fragment_links(
        self, fragment_id: str, statement_ids: List[str]
    ) -> None:
        """Insert link rows mapping a fragment to its ordered statements. Idempotent."""
        sql = """
            INSERT OR IGNORE INTO fragment_statement_links
                (fragment_id, statement_id, position_in_fragment)
            VALUES (:fragment_id, :statement_id, :position_in_fragment)
        """
        rows = [
            {
                "fragment_id": fragment_id,
                "statement_id": sid,
                "position_in_fragment": pos,
            }
            for pos, sid in enumerate(statement_ids)
        ]
        if rows:
            with self._connect() as conn:
                conn.executemany(sql, rows)

    # ------------------------------------------------------------------
    # Topic summaries
    # ------------------------------------------------------------------

    def insert_topic_summary(self, data: Dict[str, Any]) -> str:
        """Insert a topic summary row. Idempotent on (conversation_id, topic_index).

        Returns the topic_id.
        """
        sql = """
            INSERT OR IGNORE INTO topic_summaries (
                topic_id, conversation_id, topic_index, topic_segment_range,
                summary, bullets, action_items, lang, status, warnings,
                provider, model, latency_ms, created_at
            ) VALUES (
                :topic_id, :conversation_id, :topic_index, :topic_segment_range,
                :summary, :bullets, :action_items, :lang, :status, :warnings,
                :provider, :model, :latency_ms, :created_at
            )
        """
        conv_id = data["parent_conversation_id"]
        topic_idx = data["topic_index"]
        topic_id = f"{conv_id}_t{topic_idx:03d}"
        row = {
            "topic_id": topic_id,
            "conversation_id": conv_id,
            "topic_index": topic_idx,
            "topic_segment_range": data.get("topic_segment_range"),
            "summary": data.get("summary") or "",
            "bullets": json.dumps(data.get("bullets") or [], ensure_ascii=False),
            "action_items": json.dumps(data.get("action_items") or [], ensure_ascii=False),
            "lang": data.get("lang"),
            "status": data.get("status") or "ok",
            "warnings": json.dumps(data.get("warnings", []), ensure_ascii=False)
            if data.get("warnings") else None,
            "provider": data.get("provider") or "local",
            "model": data.get("model") or "",
            "latency_ms": data.get("latency_ms"),
            "created_at": data.get("created_at") or _now_iso(),
        }
        with self._connect() as conn:
            conn.execute(sql, row)
        return topic_id

    def insert_topic_segments(self, topic_id: str, segment_ids: List[str]) -> None:
        """Insert topic_segments mapping rows. Idempotent."""
        sql = """
            INSERT OR IGNORE INTO topic_segments (topic_id, segment_id, position)
            VALUES (:topic_id, :segment_id, :position)
        """
        rows = [
            {"topic_id": topic_id, "segment_id": sid, "position": pos}
            for pos, sid in enumerate(segment_ids)
        ]
        if rows:
            with self._connect() as conn:
                conn.executemany(sql, rows)

    def delete_topic_summaries(self, conversation_id: str) -> None:
        """Delete all topic summary rows (and cascade topic_segments) for a conversation."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM topic_summaries WHERE conversation_id = ?",
                (conversation_id,),
            )

    # ------------------------------------------------------------------
    # Retrieval reconstruction
    # ------------------------------------------------------------------

    def get_fragments_with_statements(
        self, fragment_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch full fragment records hydrated from the relational chain.

        For each fragment, returns:
          - fragment metadata
          - ordered statements (via fragment_statement_links)
          - parent extract metadata
          - parent segment metadata (including segment_text)
          - parent conversation metadata

        Input order is preserved. Raises nothing — missing IDs are silently
        omitted (matches get_by_ids pattern).
        """
        if not fragment_ids:
            return []

        placeholders = ",".join("?" * len(fragment_ids))
        sql = f"""
            SELECT
                f.fragment_id, f.extract_id, f.fragment_index, f.title,
                f.retrieval_text, f.status AS fragment_status,
                f.qdrant_point_id, f.embedded_at, f.embedding_model,
                f.created_at AS fragment_created_at,
                e.segment_id, e.segment_index, e.parent_conversation_id,
                e.provider, e.model, e.status AS extract_status,
                s.segment_text, s.conversation_date, s.start_position, s.end_position,
                c.title AS conversation_title, c.source_platform,
                fsl.statement_id, fsl.position_in_fragment,
                st.statement_index, st.speaker, st.text AS statement_text
            FROM fragments f
            JOIN extracts e ON f.extract_id = e.extract_id
            JOIN segments s ON e.segment_id = s.segment_id
            JOIN conversations c ON e.parent_conversation_id = c.conversation_id
            LEFT JOIN fragment_statement_links fsl ON fsl.fragment_id = f.fragment_id
            LEFT JOIN extract_statements st ON st.statement_id = fsl.statement_id
            WHERE f.fragment_id IN ({placeholders})
            ORDER BY f.fragment_id, fsl.position_in_fragment ASC
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, fragment_ids).fetchall()

        # Assemble: one dict per fragment_id
        fragments: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            fid = row["fragment_id"]
            if fid not in fragments:
                fragments[fid] = {
                    "fragment_id": fid,
                    "extract_id": row["extract_id"],
                    "fragment_index": row["fragment_index"],
                    "title": row["title"],
                    "retrieval_text": row["retrieval_text"],
                    "status": row["fragment_status"],
                    "qdrant_point_id": row["qdrant_point_id"],
                    "embedded_at": row["embedded_at"],
                    "embedding_model": row["embedding_model"],
                    "created_at": row["fragment_created_at"],
                    "segment_id": row["segment_id"],
                    "segment_index": row["segment_index"],
                    "parent_conversation_id": row["parent_conversation_id"],
                    "provider": row["provider"],
                    "model": row["model"],
                    "segment_text": row["segment_text"],
                    "conversation_date": row["conversation_date"],
                    "conversation_title": row["conversation_title"],
                    "source_platform": row["source_platform"],
                    "statements": [],
                }
            if row["statement_id"] is not None:
                fragments[fid]["statements"].append({
                    "statement_id": row["statement_id"],
                    "statement_index": row["statement_index"],
                    "speaker": row["speaker"],
                    "text": row["statement_text"],
                    "position_in_fragment": row["position_in_fragment"],
                })

        # Preserve input order
        return [fragments[fid] for fid in fragment_ids if fid in fragments]

    # ------------------------------------------------------------------
    # Job tracking
    # ------------------------------------------------------------------

    def create_job(self, job_type: str, input_metadata: Dict[str, Any]) -> str:
        """Create a new job row in 'pending' status. Returns the job_id."""
        job_id = f"job_{uuid4().hex[:12]}"
        now = _now_iso()
        sql = """
            INSERT INTO jobs (job_id, job_type, status, input_metadata, created_at)
            VALUES (?, ?, 'pending', ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (job_id, job_type, json.dumps(input_metadata), now))
        return job_id

    def mark_job_running(self, job_id: str) -> None:
        sql = "UPDATE jobs SET status='running', started_at=? WHERE job_id=?"
        with self._connect() as conn:
            conn.execute(sql, (_now_iso(), job_id))

    def mark_job_succeeded(self, job_id: str, result: Dict[str, Any]) -> None:
        sql = "UPDATE jobs SET status='succeeded', result=?, finished_at=? WHERE job_id=?"
        with self._connect() as conn:
            conn.execute(sql, (json.dumps(result), _now_iso(), job_id))

    def mark_job_failed(self, job_id: str, error: str) -> None:
        sql = "UPDATE jobs SET status='failed', error=?, finished_at=? WHERE job_id=?"
        with self._connect() as conn:
            conn.execute(sql, (error, _now_iso(), job_id))

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM jobs WHERE job_id=?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (job_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["input_metadata"] = json.loads(d["input_metadata"]) if d["input_metadata"] else {}
        d["result"] = json.loads(d["result"]) if d["result"] else None
        return d

    def list_jobs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (limit, offset)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["input_metadata"] = json.loads(d["input_metadata"]) if d["input_metadata"] else {}
            d["result"] = json.loads(d["result"]) if d["result"] else None
            result.append(d)
        return result

    def count_jobs(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Web read-only helpers (no writes, no Qdrant)
    # ------------------------------------------------------------------

    def count_all(self) -> Dict[str, int]:
        """Return per-table row counts for the dashboard."""
        tables = [
            "source_files", "conversations", "segments",
            "extracts", "fragments", "extract_statements", "jobs",
        ]
        counts: Dict[str, int] = {}
        with self._connect() as conn:
            for table in tables:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0] if row else 0
        return counts

    def list_source_files(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return source file rows, newest first."""
        sql = "SELECT * FROM source_files ORDER BY created_at DESC LIMIT ? OFFSET ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def get_source_file(self, source_file_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single source file row by ID."""
        sql = "SELECT * FROM source_files WHERE source_file_id = ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (source_file_id,)).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Return conversation rows, newest first."""
        sql = "SELECT * FROM conversations ORDER BY created_at DESC LIMIT ? OFFSET ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single conversation row by ID."""
        sql = "SELECT * FROM conversations WHERE conversation_id = ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (conversation_id,)).fetchone()
        return dict(row) if row else None

    def list_conversations_for_source(self, source_file_id: str) -> List[Dict[str, Any]]:
        """Return conversations linked to a source file (raw or normalized)."""
        sql = """
            SELECT * FROM conversations
            WHERE raw_source_file_id = ? OR normalized_source_file_id = ?
            ORDER BY created_at DESC
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (source_file_id, source_file_id)).fetchall()
        return [dict(r) for r in rows]

    def list_segments_for_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Return segment rows for a conversation, ordered by segment_index."""
        sql = """
            SELECT * FROM segments
            WHERE conversation_id = ?
            ORDER BY segment_index ASC
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (conversation_id,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["message_ids"] = json.loads(d.get("message_ids_json") or "[]")
            result.append(d)
        return result

    def get_extract(self, extract_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single extract row by ID."""
        sql = "SELECT * FROM extracts WHERE extract_id = ?"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (extract_id,)).fetchone()
        if row is None:
            return None
        r = dict(row)
        r["warnings"] = json.loads(r["warnings"]) if r["warnings"] else []
        return r

    def list_fragments_for_extract(self, extract_id: str) -> List[Dict[str, Any]]:
        """Return fragment rows for an extract, ordered by fragment_index."""
        sql = """
            SELECT * FROM fragments
            WHERE extract_id = ?
            ORDER BY fragment_index ASC
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (extract_id,)).fetchall()
        return [dict(r) for r in rows]

    def count_extracts_for_conversation(self, conversation_id: str) -> int:
        """Return the number of extract rows for a conversation."""
        sql = "SELECT COUNT(*) FROM extracts WHERE parent_conversation_id = ?"
        with self._connect() as conn:
            row = conn.execute(sql, (conversation_id,)).fetchone()
        return row[0] if row else 0

    def recent_records(self, n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Return the n most-recent rows per entity table for the dashboard."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        queries = {
            "source_files": "SELECT source_file_id AS id, original_filename AS label, "
                            "created_at FROM source_files ORDER BY created_at DESC LIMIT ?",
            "conversations": "SELECT conversation_id AS id, COALESCE(title, conversation_id) "
                             "AS label, created_at FROM conversations "
                             "ORDER BY created_at DESC LIMIT ?",
            "segments": "SELECT segment_id AS id, "
                        "('Seg #' || segment_index || ' — ' || conversation_id) AS label, "
                        "created_at FROM segments ORDER BY created_at DESC LIMIT ?",
            "extracts": "SELECT extract_id AS id, extract_id AS label, "
                        "created_at FROM extracts ORDER BY created_at DESC LIMIT ?",
            "fragments": "SELECT fragment_id AS id, fragment_id AS label, "
                         "created_at FROM fragments ORDER BY created_at DESC LIMIT ?",
        }
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            for key, sql in queries.items():
                rows = conn.execute(sql, (n,)).fetchall()
                result[key] = [dict(r) for r in rows]
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)
            conn.execute(
                "INSERT OR REPLACE INTO _jarvis_meta (key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
        logger.debug(f"SQLite store ready at {self.db_path}")
