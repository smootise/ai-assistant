"""Email migrations use temporary synthetic databases, including live WAL data."""

from contextlib import closing
from pathlib import Path
import sqlite3
from unittest.mock import patch

import pytest

from jarvis import email_store
from jarvis.email_store import EmailError, EmailStore, fingerprint, migrate_email_schema
from jarvis.store import SummaryStore
from tests.test_store_compatibility import create_existing


def baseline(path: Path) -> str:
    with closing(email_store.connect(path, read_only=True)) as connection:
        return fingerprint(connection)


def test_explicit_migration_and_backup_preserve_existing_data(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    create_existing(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript("CREATE TABLE extra (id INTEGER PRIMARY KEY, content BLOB);")
        connection.execute("INSERT INTO extra VALUES (1, ?)", (b"synthetic bytes",))
        connection.commit()
    before = baseline(path)
    assert migrate_email_schema(path, backup, "mom") == "installed"
    assert baseline(path) == before == baseline(backup)
    with closing(email_store.connect(backup, read_only=True)) as connection:
        assert (
            connection.execute(
                "SELECT value FROM _jarvis_meta WHERE key='email_schema_version'"
            ).fetchone()
            is None
        )
    with closing(EmailStore(path, "mom", read_only=True)) as store:
        assert store.status()["messages"] == {}
    SummaryStore(str(path), read_only=True)
    assert migrate_email_schema(path, tmp_path / "unused.db", "mom") == "already_installed"
    assert not (tmp_path / "unused.db").exists()


def test_migration_includes_committed_wal_data(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    SummaryStore(str(path))
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE preserve (text TEXT)")
        writer.execute("INSERT INTO preserve VALUES ('committed WAL record')")
        writer.commit()
        backup = tmp_path / "backup.db"
        migrate_email_schema(path, backup, "mom")
        with closing(email_store.connect(backup, read_only=True)) as restored:
            assert restored.execute("SELECT * FROM preserve").fetchall() == [
                ("committed WAL record",)
            ]


def test_no_automatic_email_migration(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    create_existing(path)
    before = baseline(path)
    with pytest.raises(EmailError, match="not installed"):
        EmailStore(path, "mom")
    assert baseline(path) == before


def test_migration_rolls_back_ddl_and_metadata_on_validation_failure(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    create_existing(path)
    before = baseline(path)
    with patch.object(
        email_store, "validate_email_schema", side_effect=EmailError("Injected validation failure")
    ):
        with pytest.raises(EmailError):
            migrate_email_schema(path, tmp_path / "backup.db", "mom")
    assert baseline(path) == before
    with closing(email_store.connect(path, read_only=True)) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'email_%'"
            ).fetchall()
            == []
        )
        assert (
            connection.execute(
                "SELECT value FROM _jarvis_meta WHERE key='email_schema_version'"
            ).fetchone()
            is None
        )


def test_stale_backup_prevents_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "source.db"
    create_existing(path)
    original = email_store.connect

    def connect_with_new_write(target: Path, *, read_only: bool = False) -> sqlite3.Connection:
        if target == path and not read_only:
            with closing(original(path)) as writer:
                writer.execute("INSERT INTO _jarvis_meta VALUES ('concurrent_write', 'preserved')")
                writer.commit()
        return original(target, read_only=read_only)

    monkeypatch.setattr(email_store, "connect", connect_with_new_write)
    with pytest.raises(EmailError, match="changed after backup"):
        migrate_email_schema(path, tmp_path / "backup.db", "mom")
    with closing(original(path, read_only=True)) as connection:
        assert connection.execute(
            "SELECT value FROM _jarvis_meta WHERE key='concurrent_write'"
        ).fetchone() == ("preserved",)
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='email_items'").fetchone()
            is None
        )


def test_backup_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    create_existing(path)
    backup = tmp_path / "backup.db"
    backup.write_bytes(b"preserve this backup")
    with pytest.raises(EmailError, match="new, unredirected"):
        migrate_email_schema(path, backup, "mom")
    assert backup.read_bytes() == b"preserve this backup"


def test_profile_binding_and_incomplete_extension_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    create_existing(path)
    migrate_email_schema(path, tmp_path / "backup.db", "mom")
    with pytest.raises(EmailError, match="another profile"):
        EmailStore(path, "personal")
    with closing(email_store.connect(path)) as connection:
        connection.execute("DROP INDEX idx_email_items_archive_status")
        connection.commit()
    with pytest.raises(EmailError, match="incomplete"):
        EmailStore(path, "mom")


def test_unversioned_partial_extension_is_not_repaired(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    create_existing(path)
    with closing(email_store.connect(path)) as connection:
        connection.execute(email_store.EMAIL_DDL[0])
        connection.commit()
    with pytest.raises(EmailError, match="Unversioned"):
        migrate_email_schema(path, tmp_path / "backup.db", "mom")
    assert not (tmp_path / "backup.db").exists()
