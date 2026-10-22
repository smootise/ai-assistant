"""Only synthetic SQLite files are used to exercise the v8 compatibility boundary."""

import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.store import DatabaseCompatibilityError, SummaryStore, _DDL, _v8_contract


def create_existing(path: Path, ddl: str = _DDL, version: str = "8") -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(ddl)
        conn.execute("INSERT INTO _jarvis_meta VALUES ('schema_version', ?)", (version,))
        conn.execute("INSERT INTO _jarvis_meta VALUES ('synthetic_extra', 'preserved')")
        conn.execute(
            "INSERT INTO conversations (conversation_id, title, source_platform, created_at) "
            "VALUES ('synthetic-conversation', 'Preserve this record', 'chatgpt', '2026-01-01')"
        )
        conn.commit()


def snapshot(path: Path) -> tuple:
    with closing(sqlite3.connect(path)) as conn:
        schema = conn.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()
        meta = conn.execute("SELECT * FROM _jarvis_meta ORDER BY key").fetchall()
        version = conn.execute("PRAGMA schema_version").fetchone()
        journal = conn.execute("PRAGMA journal_mode").fetchone()
    return path.read_bytes(), path.stat().st_mtime_ns, schema, meta, version, journal


@pytest.mark.parametrize("journal", ["DELETE", "WAL"])
def test_existing_v8_is_validated_read_only(tmp_path: Path, journal: str) -> None:
    path = tmp_path / "existing.db"
    create_existing(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"PRAGMA journal_mode={journal}")
        conn.executescript(
            "CREATE TABLE extra (value TEXT); CREATE INDEX extra_idx ON extra(value);"
        )
        conn.execute("ALTER TABLE conversations ADD COLUMN annotation TEXT")
    before = snapshot(path)
    statements = []
    connect = sqlite3.connect
    _v8_contract()  # Warm the disposable in-memory reference before tracing file access.

    def trace_connect(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        assert "mode=ro" in database
        conn = connect(database, *args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    with (
        patch("jarvis.store.sqlite3.connect", side_effect=trace_connect),
        patch.object(
            SummaryStore,
            "_init_db",
            side_effect=AssertionError("Existing DB must never initialize"),
        ),
    ):
        store = SummaryStore(str(path))
    assert snapshot(path) == before
    assert statements
    assert all(
        statement.lstrip().upper().startswith(("SELECT", "PRAGMA")) for statement in statements
    )
    assert store.count_all()["fragments"] == 0
    assert store.get_conversation("synthetic-conversation")["title"] == "Preserve this record"


def test_new_database_initializes_v8(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "new.db"
    store = SummaryStore(str(path))
    assert store.count_all()["jobs"] == 0
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT * FROM _jarvis_meta").fetchall() == [("schema_version", "8")]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    SummaryStore(str(path))


@pytest.mark.parametrize("version", ["7", "9", "", "08"])
def test_unsupported_versions_fail_unchanged(tmp_path: Path, version: str) -> None:
    path = tmp_path / "old.db"
    create_existing(path, version=version)
    before = snapshot(path)
    with pytest.raises(DatabaseCompatibilityError, match="version 8"):
        SummaryStore(str(path))
    assert snapshot(path) == before


@pytest.mark.parametrize(
    "change",
    [
        "DROP TABLE jobs",
        "ALTER TABLE fragments RENAME COLUMN retrieval_text TO wrong_name",
        "DELETE FROM _jarvis_meta WHERE key='schema_version'",
        "ALTER TABLE jobs ADD COLUMN incompatible TEXT NOT NULL",
    ],
)
def test_incomplete_schema_fails_unchanged(tmp_path: Path, change: str) -> None:
    path = tmp_path / "incomplete.db"
    create_existing(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(change)
        conn.commit()
    before = snapshot(path)
    with pytest.raises(DatabaseCompatibilityError):
        SummaryStore(str(path))
    assert snapshot(path) == before


@pytest.mark.parametrize(
    "ddl",
    [
        _DDL.replace("retrieval_text  TEXT NOT NULL", "retrieval_text  INTEGER NOT NULL"),
        _DDL.replace("UNIQUE(extract_id, fragment_index)", "CHECK(fragment_index >= 0)"),
        _DDL.replace("REFERENCES extracts(extract_id) ON DELETE CASCADE", ""),
        _DDL.replace("CHECK(status IN ('pending','running','succeeded','failed'))", ""),
    ],
)
def test_incompatible_columns_and_constraints(tmp_path: Path, ddl: str) -> None:
    path = tmp_path / "invalid.db"
    create_existing(path, ddl=ddl)
    before = snapshot(path)
    with pytest.raises(DatabaseCompatibilityError):
        SummaryStore(str(path))
    assert snapshot(path) == before


@pytest.mark.parametrize("content", [b"", b"not a SQLite database"])
def test_existing_empty_or_non_database_files_are_not_initialized(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "invalid.db"
    path.write_bytes(content)
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(DatabaseCompatibilityError):
        SummaryStore(str(path))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_orphan_sidecars_prevent_initialization(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / "missing.db"
    sidecar = Path(f"{path}{suffix}")
    sidecar.write_bytes(b"synthetic recovery data")
    before = sidecar.read_bytes(), sidecar.stat().st_mtime_ns
    with pytest.raises(DatabaseCompatibilityError, match="sidecar"):
        SummaryStore(str(path))
    assert not path.exists()
    assert (sidecar.read_bytes(), sidecar.stat().st_mtime_ns) == before


def test_removed_database_is_not_silently_recreated(tmp_path: Path) -> None:
    path = tmp_path / "temporary.db"
    store = SummaryStore(str(path))
    path.unlink()
    with pytest.raises(sqlite3.OperationalError):
        store.count_all()
    assert not path.exists()
