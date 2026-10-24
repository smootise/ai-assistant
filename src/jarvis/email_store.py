"""Explicit, backed-up email schema extension and profile-bound SQLite access."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from jarvis.pst import serialize
from jarvis.store import SummaryStore

EMAIL_SCHEMA_VERSION = "1"
EMAIL_META = {"email_schema_version", "email_profile_name"}
EMAIL_DDL = (
    """CREATE TABLE email_archives (
        archive_id TEXT PRIMARY KEY,
        size_bytes INTEGER NOT NULL,
        parser_version TEXT NOT NULL,
        integrity_status TEXT NOT NULL CHECK(integrity_status IN ('verified','source_changed')),
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE email_import_runs (
        run_id TEXT PRIMARY KEY,
        archive_id TEXT NOT NULL REFERENCES email_archives(archive_id),
        source_path TEXT NOT NULL,
        status TEXT NOT NULL,
        counts_json TEXT NOT NULL,
        error_code TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT
    )""",
    """CREATE TABLE email_items (
        email_id TEXT PRIMARY KEY,
        archive_id TEXT NOT NULL REFERENCES email_archives(archive_id),
        item_key TEXT NOT NULL,
        item_kind TEXT NOT NULL CHECK(item_kind IN ('message','folder','archive')),
        status TEXT NOT NULL CHECK(status IN ('imported','skipped','failed','resolved')),
        folder_path_json TEXT NOT NULL,
        folder_locator_json TEXT NOT NULL,
        item_index INTEGER,
        pst_identifier TEXT,
        original_body TEXT,
        body_source TEXT,
        metadata_json TEXT NOT NULL,
        body_json TEXT NOT NULL,
        headers_json TEXT NOT NULL,
        recipients_json TEXT NOT NULL,
        warnings_json TEXT NOT NULL,
        error_code TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(archive_id, item_key)
    )""",
    "CREATE INDEX idx_email_items_archive_status ON email_items(archive_id, status)",
    "CREATE INDEX idx_email_runs_archive ON email_import_runs(archive_id, started_at)",
)


class EmailError(ValueError):
    """Only fixed, safe operator messages should be passed to this exception."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> str:
    return json.dumps(serialize(value), ensure_ascii=True, sort_keys=True)


def connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    mode = "ro" if read_only else "rw"
    connection = sqlite3.connect(f"{path.as_uri()}?mode={mode}", uri=True, timeout=30)
    connection.execute("PRAGMA foreign_keys=ON")
    if read_only:
        connection.execute("PRAGMA query_only=ON")
    return connection


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@lru_cache(maxsize=1)
def email_objects() -> dict[str, str]:
    with closing(sqlite3.connect(":memory:")) as connection:
        for statement in EMAIL_DDL:
            connection.execute(statement)
        return dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))


def validate_email_schema(connection: sqlite3.Connection, profile: str) -> None:
    metadata = dict(connection.execute("SELECT key, value FROM _jarvis_meta"))
    if metadata.get("email_schema_version") != EMAIL_SCHEMA_VERSION:
        raise EmailError("Email schema is not installed; run email migrate with a backup path")
    if metadata.get("email_profile_name") != profile:
        raise EmailError("Email schema belongs to another profile")
    actual = dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL"))
    for name, expected in email_objects().items():
        if "".join((actual.get(name) or "").lower().split()) != "".join(expected.lower().split()):
            raise EmailError("Email schema is incomplete or incompatible; no automatic repair")


def fingerprint(connection: sqlite3.Connection) -> str:
    """Logical baseline including arbitrary existing tables, constraints, indexes and rows."""
    digest = hashlib.sha256()
    excluded = email_objects()
    objects = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    for entry in objects:
        if entry[1] in excluded or entry[2] in excluded:
            continue
        digest.update(json_value(entry).encode("utf-8"))
        if entry[0] != "table":
            continue
        table = quote(entry[1])
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        ordering = ", ".join(quote(name) for name in columns)
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY {ordering}"):
            if entry[1] == "_jarvis_meta" and row[0] in EMAIL_META:
                continue
            digest.update(json_value(row).encode("utf-8"))
    for pragma in ("user_version", "application_id"):
        digest.update(json_value(connection.execute(f"PRAGMA {pragma}").fetchone()).encode())
    return digest.hexdigest()


def integrity_check(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise EmailError("SQLite integrity verification failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise EmailError("SQLite foreign-key verification failed")


def migrate_email_schema(db_path: Path, backup_path: Path, profile: str) -> str:
    db_path = db_path.resolve()
    SummaryStore(str(db_path), read_only=True)
    with closing(connect(db_path, read_only=True)) as source:
        markers = dict(source.execute("SELECT key, value FROM _jarvis_meta"))
        existing = {row[0] for row in source.execute("SELECT name FROM sqlite_master")}
        if "email_schema_version" in markers:
            validate_email_schema(source, profile)
            return "already_installed"
        if existing & set(email_objects()) or EMAIL_META & set(markers):
            raise EmailError("Unversioned email objects exist; explicit operator recovery required")
        backup_path = backup_path.expanduser().absolute()
        if backup_path.resolve() != backup_path or backup_path.exists() or backup_path.is_symlink():
            raise EmailError("Backup must be a new, unredirected file")
        repo = Path(__file__).resolve().parents[2]
        if backup_path.is_relative_to(repo) or any(
            (parent / ".git").exists() for parent in backup_path.parents
        ):
            raise EmailError("Backups must be outside Git repositories")
        reserved = [Path(f"{db_path}{suffix}") for suffix in ("", "-wal", "-shm", "-journal")]
        if backup_path in reserved:
            raise EmailError("Backup collides with the database or a sidecar")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with backup_path.open("xb"):
            pass
        source.execute("BEGIN")
        integrity_check(source)
        baseline = fingerprint(source)
        with closing(connect(backup_path)) as backup:
            source.backup(backup)
            integrity_check(backup)
            if fingerprint(backup) != baseline:
                raise EmailError("Backup does not match the source snapshot")
        source.rollback()
    # Reopen the backup independently, confirming committed data can be restored/read.
    SummaryStore(str(backup_path), read_only=True)
    with closing(connect(backup_path, read_only=True)) as backup:
        integrity_check(backup)
        if fingerprint(backup) != baseline:
            raise EmailError("Reopened backup verification failed")
    with closing(connect(db_path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            if fingerprint(connection) != baseline:
                raise EmailError("Database changed after backup; create a fresh backup and retry")
            for statement in EMAIL_DDL:
                connection.execute(statement)
            connection.executemany(
                "INSERT INTO _jarvis_meta(key, value) VALUES (?, ?)",
                [("email_schema_version", EMAIL_SCHEMA_VERSION), ("email_profile_name", profile)],
            )
            validate_email_schema(connection, profile)
            integrity_check(connection)
            if fingerprint(connection) != baseline:
                raise EmailError("Migration altered existing data; rolled back")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    with closing(connect(db_path, read_only=True)) as migrated:
        validate_email_schema(migrated, profile)
        integrity_check(migrated)
    return "installed"


class EmailStore:
    def __init__(self, db_path: Path, profile: str, *, read_only: bool = False) -> None:
        self.db_path = db_path.resolve()
        SummaryStore(str(self.db_path), read_only=True)
        self.connection = connect(self.db_path, read_only=read_only)
        try:
            validate_email_schema(self.connection, profile)
        except BaseException:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def require_archive_compatible(self, archive_id: str, version: str) -> None:
        archive = self.connection.execute(
            "SELECT integrity_status, parser_version FROM email_archives WHERE archive_id=?",
            (archive_id,),
        ).fetchone()
        if archive is not None and archive != ("verified", version):
            raise EmailError("Archive is quarantined or parser version changed; review required")

    def start_run(
        self,
        archive_id: str,
        size: int,
        version: str,
        source: Path,
        run_id: str,
        counts: dict[str, int],
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO email_archives VALUES (?, ?, ?, 'verified', ?) "
                "ON CONFLICT(archive_id) DO NOTHING",
                (archive_id, size, version, now()),
            )
            self.require_archive_compatible(archive_id, version)
            self.connection.execute(
                "INSERT INTO email_import_runs VALUES (?, ?, ?, 'running', ?, NULL, ?, NULL)",
                (run_id, archive_id, str(source), json_value(counts), now()),
            )

    def item_status(self, email_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT status FROM email_items WHERE email_id=?", (email_id,)
        ).fetchone()
        return row[0] if row else None

    def save_counts(self, run_id: str, counts: dict[str, int]) -> None:
        self.connection.execute(
            "UPDATE email_import_runs SET counts_json=? WHERE run_id=?",
            (json_value(counts), run_id),
        )

    def record(self, run_id: str, item: dict[str, Any], counts: dict[str, int]) -> None:
        with self.connection:
            existing = self.item_status(item["email_id"])
            if existing in ("imported", "skipped"):
                raise EmailError("Concurrent import detected; use one writer per profile")
            row = {
                **item,
                "created_at": now(),
                "updated_at": now(),
            }
            columns = list(row)
            assignments = ", ".join(
                f"{key}=excluded.{key}" for key in columns if key not in ("email_id", "created_at")
            )
            result = self.connection.execute(
                f"INSERT INTO email_items({', '.join(columns)}) "
                f"VALUES ({', '.join(':' + key for key in columns)}) "
                f"ON CONFLICT(email_id) DO UPDATE SET {assignments} "
                "WHERE email_items.status NOT IN ('imported', 'skipped')",
                row,
            )
            if result.rowcount != 1:
                raise EmailError("Concurrent import detected; original record preserved")
            self.save_counts(run_id, counts)

    def resolve_failure(self, email_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE email_items SET status='resolved', updated_at=? "
                "WHERE email_id=? AND status='failed' AND item_kind!='message'",
                (now(), email_id),
            )

    def finish_run(
        self, run_id: str, status: str, counts: dict[str, int], error: str | None
    ) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE email_import_runs SET status=?, counts_json=?, error_code=?, finished_at=? "
                "WHERE run_id=?",
                (status, json_value(counts), error, now(), run_id),
            )
            if status == "source_changed":
                self.connection.execute(
                    "UPDATE email_archives SET integrity_status='source_changed' "
                    "WHERE archive_id=(SELECT archive_id FROM email_import_runs WHERE run_id=?)",
                    (run_id,),
                )

    def status(self) -> dict[str, Any]:
        counts = {
            state: count
            for state, count in self.connection.execute(
                "SELECT status, COUNT(*) FROM email_items WHERE item_kind='message' GROUP BY status"
            )
        }
        failures = self.connection.execute(
            "SELECT COUNT(*) FROM email_items WHERE status='failed'"
        ).fetchone()[0]
        archives = [
            dict(zip(("archive_id", "size_bytes", "integrity_status"), row))
            for row in self.connection.execute(
                "SELECT archive_id, size_bytes, integrity_status FROM email_archives "
                "ORDER BY archive_id"
            )
        ]
        runs = [
            {
                "run_id": row[0],
                "archive_id": row[1],
                "status": row[2],
                "counts": json.loads(row[3]),
                "error_code": row[4],
            }
            for row in self.connection.execute(
                "SELECT run_id, archive_id, status, counts_json, error_code "
                "FROM email_import_runs ORDER BY started_at DESC LIMIT 20"
            )
        ]
        return {
            "messages": counts,
            "unresolved_failures": failures,
            "archives": archives,
            "recent_runs": runs,
        }

    def list_items(self, limit: int = 20, failures: bool = False) -> list[dict[str, Any]]:
        condition = "status='failed'" if failures else "item_kind='message' AND status='imported'"
        return [
            {"email_id": row[0], "archive_id": row[1], "status": row[2], "error_code": row[3]}
            for row in self.connection.execute(
                "SELECT email_id, archive_id, status, error_code FROM email_items "
                f"WHERE {condition} ORDER BY email_id LIMIT ?",
                (limit,),
            )
        ]

    def get_email(self, email_id: str) -> dict[str, Any] | None:
        self.connection.row_factory = sqlite3.Row
        row = self.connection.execute(
            "SELECT * FROM email_items WHERE email_id=?", (email_id,)
        ).fetchone()
        self.connection.row_factory = None
        if row is None:
            return None
        result = dict(row)
        result["archive_integrity_status"] = self.connection.execute(
            "SELECT integrity_status FROM email_archives WHERE archive_id=?",
            (result["archive_id"],),
        ).fetchone()[0]
        for name in list(result):
            if name.endswith("_json"):
                result[name[:-5]] = json.loads(result.pop(name))
        return result
