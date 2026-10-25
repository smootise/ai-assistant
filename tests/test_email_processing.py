"""All processing, migration and CLI checks use guarded temporary synthetic SQLite."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest

from jarvis import cli, email_processing, email_processing_store
from jarvis.config import load_config
from jarvis.email_cleaning import clean_email
from jarvis.email_processing import process_emails
from jarvis.email_processing_store import EmailProcessingStore, migrate_processing_schema
from jarvis.email_store import EmailError, connect, fingerprint, migrate_email_schema
from jarvis.store import SummaryStore


def seed_database(path: Path, profile: str, bodies: list[str | None]) -> None:
    SummaryStore(str(path))
    migrate_email_schema(path, path.parent / f"{profile}-before-import.db", profile)
    with closing(connect(path)) as connection:
        connection.execute(
            "INSERT INTO email_archives VALUES ('archive', 42, 'fake', 'verified', '')"
        )
        for index, body in enumerate(bodies):
            connection.execute(
                "INSERT INTO email_items VALUES (?, 'archive', ?, 'message', 'imported', "
                "'[]', '[]', ?, NULL, ?, 'plain', '{}', '{}', '{}', '[]', '[]', NULL, '', '')",
                (f"email-{index:03}", str(index), index, body),
            )
        connection.commit()


def snapshot(path: Path) -> str:
    with closing(connect(path, read_only=True)) as connection:
        return fingerprint(connection, excluded_objects={}, excluded_meta=set())


@pytest.fixture
def processing_db(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.db"
    seed_database(path, "mom", ["PRIVATE_SENTINEL: AB-123 45 €", "Hi\n-- \nName", None])
    return path


def test_migration_requires_backup_preserves_all_existing_objects_and_wal_data(
    processing_db: Path, tmp_path: Path
) -> None:
    backup = tmp_path / "before-processing.db"
    with closing(connect(processing_db)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE extra (value BLOB)")
        writer.execute("INSERT INTO extra VALUES (?)", (b"committed WAL",))
        writer.commit()
        before = snapshot(processing_db)
        assert migrate_processing_schema(processing_db, backup, "mom") == "installed"
    assert snapshot(backup) == before
    with closing(connect(processing_db, read_only=True)) as connection:
        assert (
            fingerprint(
                connection,
                excluded_objects=email_processing_store.processing_objects(),
                excluded_meta={"email_processing_schema_version"},
            )
            == before
        )
    with closing(EmailProcessingStore(processing_db, "mom", read_only=True)) as store:
        assert store.evaluate()["processed"] == 0
    assert (
        migrate_processing_schema(processing_db, tmp_path / "unused.db", "mom")
        == "already_installed"
    )
    assert not (tmp_path / "unused.db").exists()


def test_no_automatic_migration_and_wrong_profile_refused(
    processing_db: Path, tmp_path: Path
) -> None:
    before = snapshot(processing_db)
    with pytest.raises(EmailError, match="not installed"):
        EmailProcessingStore(processing_db, "mom")
    with pytest.raises(EmailError, match="another profile"):
        migrate_processing_schema(processing_db, tmp_path / "unused.db", "personal")
    assert snapshot(processing_db) == before
    assert not (tmp_path / "unused.db").exists()


def test_processing_migration_rolls_back_on_verification_failure(
    processing_db: Path, tmp_path: Path
) -> None:
    before = snapshot(processing_db)
    with patch.object(
        email_processing_store, "validate_processing_schema", side_effect=EmailError("fail")
    ):
        with pytest.raises(EmailError):
            migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    assert snapshot(processing_db) == before
    assert snapshot(tmp_path / "backup.db") == before


def test_resume_force_and_original_metadata_immutability(
    processing_db: Path, tmp_path: Path
) -> None:
    migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    with closing(connect(processing_db, read_only=True)) as connection:
        original = connection.execute("SELECT * FROM email_items ORDER BY email_id").fetchall()
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        assert process_emails(store, limit=1)["processed"] == 1
        first = store.compare("email-000")
        assert process_emails(store)["processed"] == 2
        assert process_emails(store)["attempted"] == 0
        assert store.compare("email-000") == first
        assert process_emails(store, limit=1, force=True)["processed"] == 1
        second = store.compare("email-000")
        assert second["processed_text"] == first["processed_text"]
        assert second["processed_at"] != first["processed_at"]
        report = store.evaluate()
        assert report["processed"] == 3
        assert report["manual_review"] == 1
        assert report["section_percentages"]["signature"] > 0
        assert "PRIVATE_SENTINEL" not in json.dumps(report)
    with closing(connect(processing_db, read_only=True)) as connection:
        assert (
            connection.execute("SELECT * FROM email_items ORDER BY email_id").fetchall() == original
        )


def test_interruption_and_failure_retry_preserve_completed_work(
    processing_db: Path, tmp_path: Path
) -> None:
    migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        with patch.object(
            email_processing, "clean_email", side_effect=[clean_email("one"), KeyboardInterrupt()]
        ):
            result = process_emails(store)
        assert result["status"] == "interrupted"
        assert result["processed"] == 1
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        with patch.object(
            email_processing,
            "clean_email",
            side_effect=[ValueError("PRIVATE_FAILURE"), clean_email(None)],
        ):
            result = process_emails(store)
        assert result["status"] == "partial"
        assert result["processed"] == result["failed"] == 1
        assert "PRIVATE_FAILURE" not in json.dumps(store.compare("email-001"))
        assert store.evaluate()["failed_attempts"] == 1
        assert process_emails(store)["processed"] == 1
        assert store.evaluate()["failed_attempts"] == 0
        assert store.evaluate()["processed"] == 3


def test_force_failure_keeps_previous_success_and_retries_on_resume(
    processing_db: Path, tmp_path: Path
) -> None:
    migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        process_emails(store, limit=1)
        before = store.compare("email-000")
        with patch.object(email_processing, "clean_email", side_effect=ValueError("PRIVATE")):
            assert process_emails(store, limit=1, force=True)["failed"] == 1
        failed = store.compare("email-000")
        assert failed["processed_text"] == before["processed_text"]
        assert failed["processed_at"] == before["processed_at"]
        assert failed["error_code"] == "processing_failed"
        assert process_emails(store, limit=1)["processed"] == 1
        assert store.compare("email-000")["error_code"] is None


def test_new_algorithm_version_keeps_prior_results(
    processing_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        process_emails(store)
        monkeypatch.setattr(email_processing_store, "PROCESSING_VERSION", "synthetic-v2")
        assert process_emails(store)["processed"] == 3
        assert store.connection.execute("SELECT COUNT(*) FROM email_processed").fetchone() == (6,)


def test_hard_stop_during_persistence_resumes_without_false_completion(
    processing_db: Path, tmp_path: Path
) -> None:
    migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        with patch.object(store, "save_result", side_effect=KeyboardInterrupt):
            assert process_emails(store)["status"] == "interrupted"
        assert store.evaluate()["processed"] == 0
        assert process_emails(store)["processed"] == 3


def test_failed_sqlite_write_rolls_back_and_leaves_originals_recoverable(
    processing_db: Path, tmp_path: Path
) -> None:
    migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        store.connection.execute(
            "CREATE TRIGGER fail_processing AFTER INSERT ON email_processed "
            "BEGIN SELECT RAISE(ABORT, 'synthetic storage failure'); END"
        )
        store.connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            process_emails(store)
        assert store.evaluate()["processed"] == 0
        assert store.connection.execute("SELECT COUNT(*) FROM email_items").fetchone() == (3,)
        store.connection.execute("DROP TRIGGER fail_processing")
        store.connection.commit()
        assert process_emails(store)["processed"] == 3


def test_evaluation_character_weighting_averages_and_read_only_comparison(
    processing_db: Path, tmp_path: Path
) -> None:
    migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        process_emails(store)
    before = snapshot(processing_db)
    with closing(EmailProcessingStore(processing_db, "mom", read_only=True)) as store:
        result = store.evaluate()
        assert result["average_original_length"] == round(
            (len("PRIVATE_SENTINEL: AB-123 45 €") + len("Hi\n-- \nName")) / 3, 2
        )
        assert result["average_processed_length"] == round(
            (len("PRIVATE_SENTINEL: AB-123 45 €") + len("Hi")) / 3, 2
        )
        assert sum(result["section_percentages"].values()) == pytest.approx(100, abs=0.02)
        assert store.compare("email-001")["original_body"] == "Hi\n-- \nName"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.save_result("email-001", "Hi", clean_email("Hi"))
    assert snapshot(processing_db) == before


def test_partial_processing_schema_is_not_repaired(processing_db: Path, tmp_path: Path) -> None:
    with closing(connect(processing_db)) as connection:
        connection.execute("CREATE TABLE email_processed (email_id TEXT)")
        connection.commit()
    before = snapshot(processing_db)
    with pytest.raises(EmailError, match="Unversioned"):
        migrate_processing_schema(processing_db, tmp_path / "unused.db", "mom")
    assert not (tmp_path / "unused.db").exists()
    assert snapshot(processing_db) == before


def test_quarantined_archives_and_non_imported_items_are_skipped(
    processing_db: Path, tmp_path: Path
) -> None:
    migrate_processing_schema(processing_db, tmp_path / "backup.db", "mom")
    with closing(EmailProcessingStore(processing_db, "mom")) as store:
        store.connection.execute(
            "UPDATE email_items SET status='skipped' WHERE email_id='email-000'"
        )
        store.connection.commit()
        assert process_emails(store)["processed"] == 2
        store.connection.execute("UPDATE email_archives SET integrity_status='source_changed'")
        store.connection.commit()
        assert process_emails(store, force=True)["attempted"] == 0


def test_cli_profile_isolation_compare_evaluation_and_no_services(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    databases = {}
    for profile in ("mom", "personal"):
        path = Path(load_config(profile)["db_path"])
        seed_database(path, profile, [f"PRIVATE_{profile}: REF-789 123 €"])
        databases[profile] = path
    personal_before = snapshot(databases["personal"])

    def command(*args: str) -> tuple[int, str]:
        monkeypatch.setattr(sys, "argv", ["jarvis", "--profile", "mom", "email", *args])
        code = cli.main()
        captured = capsys.readouterr()
        return code, captured.out + captured.err

    with (
        patch("jarvis.clients.OllamaClient", side_effect=AssertionError("No inference")),
        patch("jarvis.clients.VectorStore", side_effect=AssertionError("No vectors")),
        patch("jarvis.clients.EmbeddingClient", side_effect=AssertionError("No embeddings")),
        patch("jarvis.ingest.email.load_parser", side_effect=AssertionError("No PST reads")),
    ):
        assert command("process")[0] == 1
        assert command("processing-migrate", "--backup", str(profile_repo / "backup.db"))[0] == 0
        code, output = command("process", "--limit", "1")
        assert code == 0 and '"processed": 1' in output and "PRIVATE" not in output
        assert command("process")[0] == 0
        code, output = command("evaluate")
        assert code == 0 and '"processed": 1' in output and "PRIVATE" not in output
        code, output = command("compare", "email-000")
        assert code == 0 and "PRIVATE_mom" in output and "PRIVATE_personal" not in output
        assert command("compare", "not-found")[0] == 1
        with patch.object(email_processing, "clean_email", side_effect=KeyboardInterrupt):
            assert command("process", "--force")[0] == 130
        with patch.object(email_processing, "clean_email", side_effect=ValueError("PRIVATE_ERROR")):
            code, output = command("process", "--force")
            assert code == 1 and "PRIVATE_ERROR" not in output
    assert snapshot(databases["personal"]) == personal_before


def test_missing_database_is_never_created(tmp_path: Path) -> None:
    path = tmp_path / "absent.db"
    with pytest.raises(Exception):
        EmailProcessingStore(path, "mom")
    assert not path.exists()
