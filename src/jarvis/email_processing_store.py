"""Versioned derived email representations; originals are never written here."""

from __future__ import annotations

from contextlib import closing
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from jarvis.email_cleaning import CleanedEmail, KINDS, PROCESSING_VERSION
from jarvis.email_store import (
    EmailError,
    EmailStore,
    json_value,
    migrate_extension,
    now,
    validate_email_schema,
)

PROCESSING_SCHEMA_VERSION = "1"
PROCESSING_DDL = (
    """CREATE TABLE email_processed (
        email_id TEXT NOT NULL REFERENCES email_items(email_id),
        processing_version TEXT NOT NULL,
        original_sha256 TEXT NOT NULL,
        original_length INTEGER NOT NULL,
        normalized_text TEXT,
        processed_text TEXT,
        sections_json TEXT,
        indicators_json TEXT,
        needs_review INTEGER NOT NULL CHECK(needs_review IN (0,1)),
        processed_at TEXT,
        attempted_at TEXT NOT NULL,
        error_code TEXT CHECK(error_code IS NULL OR error_code='processing_failed'),
        PRIMARY KEY(email_id, processing_version)
    )""",
)


@lru_cache(maxsize=1)
def processing_objects() -> dict[str, str]:
    with closing(sqlite3.connect(":memory:")) as connection:
        for statement in PROCESSING_DDL:
            connection.execute(statement)
        return dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))


def validate_processing_schema(connection: sqlite3.Connection, profile: str) -> None:
    validate_email_schema(connection, profile)
    markers = dict(connection.execute("SELECT key, value FROM _jarvis_meta"))
    if markers.get("email_processing_schema_version") != PROCESSING_SCHEMA_VERSION:
        raise EmailError(
            "Processing schema is not installed; run email processing-migrate with backup"
        )
    actual = dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))
    for name, expected in processing_objects().items():
        if "".join((actual.get(name) or "").lower().split()) != "".join(expected.lower().split()):
            raise EmailError("Processing schema is incomplete or incompatible; no automatic repair")


def migrate_processing_schema(db_path: Path, backup_path: Path, profile: str) -> str:
    return migrate_extension(
        db_path,
        backup_path,
        profile,
        ddl=PROCESSING_DDL,
        objects=processing_objects(),
        markers={"email_processing_schema_version": PROCESSING_SCHEMA_VERSION},
        validate=validate_processing_schema,
        prerequisite=validate_email_schema,
    )


class EmailProcessingStore(EmailStore):
    def __init__(self, db_path: Path, profile: str, *, read_only: bool = False) -> None:
        super().__init__(db_path, profile, read_only=read_only)
        try:
            validate_processing_schema(self.connection, profile)
        except BaseException:
            self.close()
            raise

    def candidates(self, *, force: bool = False) -> Iterator[tuple[str, str | None]]:
        last_id: str | None = None
        while True:
            after = "AND e.email_id>? " if last_id is not None else ""
            row = self.connection.execute(
                "SELECT e.email_id, e.original_body FROM email_items e "
                "JOIN email_archives a ON a.archive_id=e.archive_id "
                "LEFT JOIN email_processed p ON p.email_id=e.email_id AND p.processing_version=? "
                "WHERE e.status='imported' AND e.item_kind='message' "
                "AND a.integrity_status='verified' "
                + after
                + "AND (? OR p.processed_at IS NULL OR p.error_code IS NOT NULL) "
                "ORDER BY e.email_id LIMIT 1",
                (PROCESSING_VERSION, *((last_id,) if last_id is not None else ()), force),
            ).fetchone()
            if row is None:
                return
            last_id = row[0]
            yield row

    def completed_count(self) -> int:
        return self.connection.execute(
            "SELECT COUNT(*) FROM email_processed p JOIN email_items e USING(email_id) "
            "JOIN email_archives a USING(archive_id) WHERE processing_version=? "
            "AND p.processed_at IS NOT NULL AND p.error_code IS NULL "
            "AND e.status='imported' AND a.integrity_status='verified'",
            (PROCESSING_VERSION,),
        ).fetchone()[0]

    def save_result(self, email_id: str, original: str | None, result: CleanedEmail | None) -> None:
        timestamp = now()
        digest = hashlib.sha256((original or "").encode("utf-8")).hexdigest()
        values = (
            email_id,
            PROCESSING_VERSION,
            digest,
            len(original or ""),
            result.normalized_text if result else None,
            result.processed_text if result else None,
            json_value(result.sections_data()) if result else None,
            json_value(result.indicators) if result else None,
            int(result.needs_review) if result else 1,
            timestamp if result else None,
            timestamp,
            None if result else "processing_failed",
        )
        # A failed forced attempt keeps the previous successful representation recoverable.
        assignments = (
            "attempted_at=excluded.attempted_at, error_code=excluded.error_code"
            if result is None
            else ", ".join(
                f"{name}=excluded.{name}"
                for name in (
                    "original_sha256",
                    "original_length",
                    "normalized_text",
                    "processed_text",
                    "sections_json",
                    "indicators_json",
                    "needs_review",
                    "processed_at",
                    "attempted_at",
                    "error_code",
                )
            )
        )
        with self.connection:
            self.connection.execute(
                "INSERT INTO email_processed VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                f"ON CONFLICT(email_id, processing_version) DO UPDATE SET {assignments}",
                values,
            )

    def compare(self, email_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT e.original_body, p.normalized_text, p.processed_text, p.sections_json, "
            "p.indicators_json, p.needs_review, p.processed_at, p.error_code "
            "FROM email_items e JOIN email_processed p USING(email_id) "
            "WHERE email_id=? AND processing_version=?",
            (email_id, PROCESSING_VERSION),
        ).fetchone()
        if row is None:
            raise EmailError("No current processing record for this email in the selected profile")
        return {
            "email_id": email_id,
            "processing_version": PROCESSING_VERSION,
            "original_body": row[0],
            "normalized_text": row[1],
            "processed_text": row[2],
            "sections": json.loads(row[3]) if row[3] else [],
            "indicators": json.loads(row[4]) if row[4] else [],
            "needs_review": bool(row[5]),
            "processed_at": row[6],
            "error_code": row[7],
        }

    def evaluate(self) -> dict[str, Any]:
        count = original = processed = review = failures = total = 0
        characters = dict.fromkeys(KINDS, 0)
        for length, text, sections, needs_review, completed, error in self.connection.execute(
            "SELECT original_length, processed_text, sections_json, needs_review, processed_at, "
            "error_code FROM email_processed WHERE processing_version=?",
            (PROCESSING_VERSION,),
        ):
            failures += int(error is not None)
            review += int(bool(needs_review) or error is not None)
            if completed is None:
                continue
            count += 1
            original += length
            processed += len(text)
            for section in json.loads(sections):
                size = len(section["text"])
                characters[section["kind"]] += size
                total += size
        return {
            "processing_version": PROCESSING_VERSION,
            "processed": count,
            "average_original_length": round(original / count, 2) if count else 0,
            "average_processed_length": round(processed / count, 2) if count else 0,
            "section_percentages": {
                kind: round(size * 100 / total, 2) if total else 0
                for kind, size in characters.items()
            },
            "manual_review": review,
            "failed_attempts": failures,
        }
