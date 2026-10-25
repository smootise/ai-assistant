"""Additive preparation storage: checkpoints, shared exact text, and every occurrence."""

from __future__ import annotations

from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from jarvis.email_cleaning import PROCESSING_VERSION
from jarvis.email_preparation import (
    PreparationOptions,
    PreparedEmail,
    content_id,
    input_fingerprint,
    near_decision,
    near_features,
)
from jarvis.email_processing_store import EmailProcessingStore, validate_processing_schema
from jarvis.email_store import EmailError, json_value, migrate_extension, now

PREPARATION_SCHEMA_VERSION = "1"
NEAR_CANDIDATE_LIMIT = 200
PREPARATION_DDL = (
    """CREATE TABLE email_preparations (
        email_id TEXT NOT NULL,
        generation_key TEXT NOT NULL,
        processing_version TEXT NOT NULL,
        input_sha256 TEXT NOT NULL,
        sections_json TEXT,
        prepared_at TEXT,
        attempted_at TEXT NOT NULL,
        error_code TEXT CHECK(error_code IS NULL OR error_code='preparation_failed'),
        PRIMARY KEY(email_id, generation_key),
        FOREIGN KEY(email_id, processing_version)
            REFERENCES email_processed(email_id, processing_version)
    )""",
    """CREATE TABLE email_embedding_contents (
        generation_key TEXT NOT NULL,
        content_id TEXT NOT NULL,
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL,
        first_anchor TEXT NOT NULL,
        last_anchor TEXT NOT NULL,
        PRIMARY KEY(generation_key, content_id)
    )""",
    "CREATE INDEX idx_email_content_first ON email_embedding_contents "
    "(generation_key, first_anchor, token_count)",
    "CREATE INDEX idx_email_content_last ON email_embedding_contents "
    "(generation_key, last_anchor, token_count)",
    """CREATE TABLE email_embedding_occurrences (
        email_id TEXT NOT NULL,
        generation_key TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content_id TEXT NOT NULL,
        section_index INTEGER NOT NULL,
        start INTEGER NOT NULL,
        end INTEGER NOT NULL,
        hard_split INTEGER NOT NULL CHECK(hard_split IN (0,1)),
        transformations_json TEXT NOT NULL,
        near_json TEXT NOT NULL,
        PRIMARY KEY(email_id, generation_key, chunk_index),
        FOREIGN KEY(email_id, generation_key)
            REFERENCES email_preparations(email_id, generation_key),
        FOREIGN KEY(generation_key, content_id)
            REFERENCES email_embedding_contents(generation_key, content_id)
    )""",
    "CREATE INDEX idx_email_occurrence_content ON email_embedding_occurrences "
    "(generation_key, content_id)",
)


@lru_cache(maxsize=1)
def preparation_objects() -> dict[str, str]:
    with closing(sqlite3.connect(":memory:")) as connection:
        for statement in PREPARATION_DDL:
            connection.execute(statement)
        return dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))


def validate_preparation_schema(connection: sqlite3.Connection, profile: str) -> None:
    validate_processing_schema(connection, profile)
    markers = dict(connection.execute("SELECT key, value FROM _jarvis_meta"))
    if markers.get("email_preparation_schema_version") != PREPARATION_SCHEMA_VERSION:
        raise EmailError(
            "Preparation schema is not installed; run email preparation-migrate with backup"
        )
    actual = dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))
    for name, expected in preparation_objects().items():
        if "".join((actual.get(name) or "").lower().split()) != "".join(expected.lower().split()):
            raise EmailError(
                "Preparation schema is incomplete or incompatible; no automatic repair"
            )


def migrate_preparation_schema(db_path: Path, backup_path: Path, profile: str) -> str:
    return migrate_extension(
        db_path,
        backup_path,
        profile,
        ddl=PREPARATION_DDL,
        objects=preparation_objects(),
        markers={"email_preparation_schema_version": PREPARATION_SCHEMA_VERSION},
        validate=validate_preparation_schema,
        prerequisite=validate_processing_schema,
    )


class EmailPreparationStore(EmailProcessingStore):
    def __init__(self, db_path: Path, profile: str, *, read_only: bool = False) -> None:
        super().__init__(db_path, profile, read_only=read_only)
        try:
            validate_preparation_schema(self.connection, profile)
        except BaseException:
            self.close()
            raise

    def preparation_candidates(self, options: PreparationOptions) -> Iterator[dict[str, Any]]:
        last_id = ""
        while True:
            cursor = self.connection.execute(
                "SELECT p.email_id, p.original_sha256, p.normalized_text, p.processed_text, "
                "p.sections_json, r.input_sha256, r.prepared_at, r.error_code "
                "FROM email_processed p JOIN email_items e USING(email_id) "
                "JOIN email_archives a USING(archive_id) LEFT JOIN email_preparations r "
                "ON r.email_id=p.email_id AND r.generation_key=? "
                "WHERE p.processing_version=? AND p.processed_at IS NOT NULL "
                "AND p.error_code IS NULL AND e.status='imported' AND e.item_kind='message' "
                "AND a.integrity_status='verified' AND p.email_id>? ORDER BY p.email_id LIMIT 1",
                (options.key, PROCESSING_VERSION, last_id),
            )
            row = cursor.fetchone()
            if row is None:
                return
            last_id = row[0]
            record = dict(zip((column[0] for column in cursor.description), row))
            if row[6] is None or row[7] is not None or row[5] != input_fingerprint(record):
                yield record

    def _near_matches(self, key: str, text: str) -> dict[str, Any]:
        tokens, first, last = near_features(text)
        if len(tokens) < 12:
            return {"matches": [], "candidates_checked": 0, "candidates_truncated": False}
        candidates = self.connection.execute(
            "SELECT content_id, text FROM email_embedding_contents c WHERE generation_key=? "
            "AND (first_anchor=? OR last_anchor=?) AND token_count BETWEEN ? AND ? "
            "AND content_id!=? AND EXISTS (SELECT 1 FROM email_embedding_occurrences o "
            "WHERE o.generation_key=c.generation_key AND o.content_id=c.content_id) "
            "ORDER BY content_id LIMIT ?",
            (
                key,
                first,
                last,
                int(len(tokens) * 0.92),
                int(len(tokens) / 0.92),
                content_id(text),
                NEAR_CANDIDATE_LIMIT + 1,
            ),
        ).fetchall()
        matches = []
        for identifier, candidate in candidates[:NEAR_CANDIDATE_LIMIT]:
            decision = near_decision(text, candidate)
            if decision:
                matches.append({"content_id": identifier, **decision})
        return {
            "matches": matches,
            "candidates_checked": min(len(candidates), NEAR_CANDIDATE_LIMIT),
            "candidates_truncated": len(candidates) > NEAR_CANDIDATE_LIMIT,
        }

    def save_preparation(
        self, record: dict[str, Any], options: PreparationOptions, result: PreparedEmail | None
    ) -> None:
        timestamp = now()
        email_id = record["email_id"]
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            cursor = self.connection.execute(
                "SELECT p.original_sha256, p.normalized_text, p.processed_text, p.sections_json "
                "FROM email_processed p JOIN email_items e USING(email_id) "
                "JOIN email_archives a USING(archive_id) WHERE p.email_id=? "
                "AND p.processing_version=? AND p.processed_at IS NOT NULL "
                "AND p.error_code IS NULL AND e.status='imported' "
                "AND e.item_kind='message' AND a.integrity_status='verified'",
                (email_id, PROCESSING_VERSION),
            )
            current = cursor.fetchone()
            if current is None or input_fingerprint(
                dict(zip((column[0] for column in cursor.description), current))
            ) != input_fingerprint(record):
                raise EmailError("Preparation input changed; stop other writers and retry")
            assignment = (
                "attempted_at=excluded.attempted_at, error_code=excluded.error_code"
                if result is None
                else "input_sha256=excluded.input_sha256, sections_json=excluded.sections_json, "
                "prepared_at=excluded.prepared_at, attempted_at=excluded.attempted_at, "
                "error_code=NULL"
            )
            self.connection.execute(
                "INSERT INTO email_preparations VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                f"ON CONFLICT(email_id, generation_key) DO UPDATE SET {assignment}",
                (
                    email_id,
                    options.key,
                    PROCESSING_VERSION,
                    input_fingerprint(record),
                    json_value(result.sections) if result else None,
                    timestamp if result else None,
                    timestamp,
                    None if result else "preparation_failed",
                ),
            )
            if result is None:
                return
            self.connection.execute(
                "DELETE FROM email_embedding_occurrences WHERE email_id=? AND generation_key=?",
                (email_id, options.key),
            )
            for index, chunk in enumerate(result.chunks):
                identifier = content_id(chunk.text)
                tokens, first, last = near_features(chunk.text)
                self.connection.execute(
                    "INSERT INTO email_embedding_contents VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(generation_key, content_id) DO NOTHING",
                    (options.key, identifier, chunk.text, len(tokens), first, last),
                )
                stored = self.connection.execute(
                    "SELECT text FROM email_embedding_contents "
                    "WHERE generation_key=? AND content_id=?",
                    (options.key, identifier),
                ).fetchone()[0]
                if stored != chunk.text:
                    raise EmailError("Content identity conflict; existing text preserved")
                near = self._near_matches(options.key, chunk.text)
                self.connection.execute(
                    "INSERT INTO email_embedding_occurrences VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        email_id,
                        options.key,
                        index,
                        identifier,
                        chunk.section_index,
                        chunk.start,
                        chunk.end,
                        int(chunk.hard_split),
                        json_value(chunk.transformations),
                        json_value(near),
                    ),
                )

    def _references(self, key: str, identifier: str) -> list[dict[str, Any]]:
        return [
            dict(zip(("email_id", "chunk_index", "section_index", "start", "end"), row))
            for row in self.connection.execute(
                "SELECT email_id, chunk_index, section_index, start, end "
                "FROM email_embedding_occurrences WHERE generation_key=? AND content_id=? "
                "ORDER BY email_id, chunk_index",
                (key, identifier),
            )
        ]

    def preview_preparation(self, email_id: str, options: PreparationOptions) -> dict[str, Any]:
        cursor = self.connection.execute(
            "SELECT e.original_body, p.normalized_text, p.processed_text, p.original_sha256, "
            "p.sections_json, r.sections_json AS preparation_sections, r.input_sha256, "
            "r.prepared_at, r.attempted_at, r.error_code, p.error_code AS processing_error "
            "FROM email_preparations r JOIN email_items e USING(email_id) "
            "JOIN email_processed p ON p.email_id=r.email_id "
            "AND p.processing_version=r.processing_version "
            "WHERE r.email_id=? AND r.generation_key=?",
            (email_id, options.key),
        )
        row = cursor.fetchone()
        if row is None:
            raise EmailError(
                "No preparation record for this email and configuration in this profile"
            )
        source = dict(zip((column[0] for column in cursor.description), row))
        stale = source["input_sha256"] != input_fingerprint(source)
        sections = json.loads(source["preparation_sections"] or "[]")
        for section in sections:
            section["source_text"] = (
                None if stale else source["normalized_text"][section["start"] : section["end"]]
            )
        chunks = []
        for entry in self.connection.execute(
            "SELECT o.chunk_index, o.content_id, o.section_index, o.start, o.end, o.hard_split, "
            "c.text, o.transformations_json, o.near_json "
            "FROM email_embedding_occurrences o JOIN email_embedding_contents c "
            "USING(generation_key, content_id) WHERE o.email_id=? AND o.generation_key=? "
            "ORDER BY o.chunk_index",
            (email_id, options.key),
        ):
            chunk = dict(
                zip(
                    (
                        "chunk_index",
                        "content_id",
                        "section_index",
                        "start",
                        "end",
                        "hard_split",
                        "text",
                        "transformations",
                        "near_at_preparation",
                    ),
                    entry,
                )
            )
            chunk["transformations"] = json.loads(chunk["transformations"])
            chunk["near_at_preparation"] = json.loads(chunk["near_at_preparation"])
            chunk["references"] = self._references(options.key, chunk["content_id"])
            chunk["exact_duplicate"] = len(chunk["references"]) > 1
            # Re-evaluate advisory matches read-only so earlier chunks see later arrivals too.
            chunk["near_duplicates"] = self._near_matches(options.key, chunk["text"])
            for match in chunk["near_duplicates"]["matches"]:
                match["references"] = self._references(options.key, match["content_id"])
            chunks.append(chunk)
        return {
            "email_id": email_id,
            "generation_key": options.key,
            "original_body": source["original_body"],
            "normalized_text": source["normalized_text"],
            "processed_text": source["processed_text"],
            "sections": sections,
            "chunks": chunks,
            "prepared_at": source["prepared_at"],
            "attempted_at": source["attempted_at"],
            "error_code": source["error_code"],
            "stale_input": stale,
            "processing_error": source["processing_error"],
        }

    def evaluate_preparation(self, options: PreparationOptions) -> dict[str, Any]:
        prepared = failed = stale = original = processed = normalized = excluded = 0
        exclusions: dict[str, int] = {}
        excluded_kinds: dict[str, int] = {}
        ambiguous_sections = 0
        for row in self.connection.execute(
            "SELECT r.prepared_at, r.error_code, r.input_sha256, r.sections_json, "
            "p.original_sha256, p.normalized_text, p.processed_text, "
            "p.sections_json, p.original_length "
            "FROM email_preparations r JOIN email_processed p "
            "ON p.email_id=r.email_id AND p.processing_version=r.processing_version "
            "WHERE r.generation_key=?",
            (options.key,),
        ):
            failed += int(row[1] is not None)
            if row[0] is None:
                continue
            prepared += 1
            current = dict(
                zip(
                    ("original_sha256", "normalized_text", "processed_text", "sections_json"),
                    row[4:8],
                )
            )
            is_stale = row[2] != input_fingerprint(current)
            stale += int(is_stale)
            if is_stale:
                continue
            original += row[8]
            processed += len(row[6])
            normalized += len(row[5])
            for section in json.loads(row[3]):
                ambiguous_sections += section["rule"] in {
                    "ambiguous_signature_retained",
                    "upstream_exclusion_reconsidered",
                }
                if section["excluded"]:
                    size = section["end"] - section["start"]
                    excluded += size
                    exclusions[section["rule"]] = exclusions.get(section["rule"], 0) + size
                    kind = section["kind"]
                    excluded_kinds[kind] = excluded_kinds.get(kind, 0) + size
        totals = self.connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT content_id), COALESCE(SUM(length(c.text)), 0), "
            "COALESCE(SUM(o.hard_split), 0) FROM email_embedding_occurrences o "
            "JOIN email_embedding_contents c USING(generation_key, content_id) "
            "WHERE generation_key=?",
            (options.key,),
        ).fetchone()
        unique_chars = near_count = truncated = 0
        for text in self.connection.execute(
            "SELECT text FROM email_embedding_contents c WHERE generation_key=? "
            "AND EXISTS (SELECT 1 FROM email_embedding_occurrences o "
            "WHERE o.generation_key=c.generation_key AND o.content_id=c.content_id)",
            (options.key,),
        ):
            unique_chars += len(text[0])
            decision = self._near_matches(options.key, text[0])
            near_count += bool(decision["matches"])
            truncated += decision["candidates_truncated"]
        return {
            "generation_key": options.key,
            "prepared": prepared,
            "failed_attempts": failed,
            "stale_inputs": stale,
            "total_chunks": totals[0],
            "unique_chunks": totals[1],
            "exact_duplicate_occurrences": totals[0] - totals[1],
            "exact_duplicate_rate": (
                round((totals[0] - totals[1]) / totals[0], 6) if totals[0] else 0
            ),
            "unique_chunks_with_near_matches": near_count,
            "near_candidate_truncations": truncated,
            "original_characters": original,
            "processed_characters": processed,
            "normalized_characters": normalized,
            "excluded_characters": excluded,
            "excluded_by_rule": exclusions,
            "excluded_by_kind": excluded_kinds,
            "ambiguous_sections_retained": ambiguous_sections,
            "source_character_totals_exclude_stale": True,
            "retained_source_characters": normalized - excluded,
            "chunk_characters_with_overlap": totals[2],
            "unique_chunk_characters": unique_chars,
            "hard_split_chunks": totals[3],
            "estimated_reduction_vs_original": (
                None if stale else (round(1 - unique_chars / original, 6) if original else 0)
            ),
            "estimated_reduction_vs_processed": (
                None if stale else (round(1 - unique_chars / processed, 6) if processed else 0)
            ),
        }
