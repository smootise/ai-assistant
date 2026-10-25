"""Temporary SQLite only: backups, exact references, retry and profile-routed CLI."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest

from jarvis import cli, email_preparation, email_preparation_store
from jarvis.config import load_config
from jarvis.email_preparation import PreparationOptions, prepare_emails
from jarvis.email_preparation_store import (
    EmailPreparationStore,
    migrate_preparation_schema,
    preparation_objects,
)
from jarvis.email_processing import process_emails
from jarvis.email_processing_store import EmailProcessingStore, migrate_processing_schema
from jarvis.email_store import EmailError, connect, fingerprint
from tests.test_email_processing import seed_database, snapshot

OPTIONS = PreparationOptions()
REPEATED = "PRIVATE_SENTINEL: Livraison AB-123 confirmée à 450 € lundi matin."
NEAR = "Please confirm that the requested delivery for order AB-123 will arrive on Monday morning."


def seed_processed(path: Path, profile: str, bodies: list[str]) -> None:
    seed_database(path, profile, bodies)
    migrate_processing_schema(path, path.parent / f"{profile}-before-processing.db", profile)
    with closing(EmailProcessingStore(path, profile)) as store:
        process_emails(store)


@pytest.fixture
def processed_db(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.db"
    seed_processed(
        path,
        "synthetic",
        [
            REPEATED,
            "New fact CD-456\nOn Monday Jane wrote:\n> " + REPEATED,
            NEAR,
            NEAR.upper(),
            "Hello\nRegards\nJane Smith\nSales Manager\njane@example.test",
        ],
    )
    return path


@pytest.fixture
def prepared_db(processed_db: Path, tmp_path: Path) -> Path:
    migrate_preparation_schema(processed_db, tmp_path / "before-preparation.db", "synthetic")
    return processed_db


def test_migration_preserves_processed_original_and_arbitrary_wal_rows(
    processed_db: Path, tmp_path: Path
) -> None:
    with closing(connect(processed_db)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE extra (value BLOB)")
        writer.execute("INSERT INTO extra VALUES (?)", (b"synthetic committed WAL",))
        writer.commit()
        before = snapshot(processed_db)
        backup = tmp_path / "verified.db"
        assert migrate_preparation_schema(processed_db, backup, "synthetic") == "installed"
    assert snapshot(backup) == before
    with closing(connect(processed_db, read_only=True)) as connection:
        assert (
            fingerprint(
                connection,
                excluded_objects=preparation_objects(),
                excluded_meta={"email_preparation_schema_version"},
            )
            == before
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert (
        migrate_preparation_schema(processed_db, tmp_path / "unused.db", "synthetic")
        == "already_installed"
    )
    assert not (tmp_path / "unused.db").exists()


def test_migration_requires_processing_schema_and_no_automatic_install(
    processed_db: Path, tmp_path: Path
) -> None:
    before = snapshot(processed_db)
    with pytest.raises(EmailError, match="not installed"):
        EmailPreparationStore(processed_db, "synthetic")
    with pytest.raises(EmailError, match="another profile"):
        migrate_preparation_schema(processed_db, tmp_path / "unused.db", "other")
    assert snapshot(processed_db) == before
    missing_processing = tmp_path / "originals-only.db"
    seed_database(missing_processing, "originals", ["Hello"])
    with pytest.raises(EmailError, match="Processing schema"):
        migrate_preparation_schema(missing_processing, tmp_path / "unused.db", "originals")
    assert not (tmp_path / "unused.db").exists()


def test_migration_verification_failure_rolls_back_all_ddl_and_preserves_backup(
    processed_db: Path, tmp_path: Path
) -> None:
    before = snapshot(processed_db)
    with patch.object(
        email_preparation_store,
        "validate_preparation_schema",
        side_effect=EmailError("synthetic validation failure"),
    ):
        with pytest.raises(EmailError):
            migrate_preparation_schema(processed_db, tmp_path / "verified.db", "synthetic")
    assert snapshot(processed_db) == before
    assert snapshot(tmp_path / "verified.db") == before


def test_partial_and_incompatible_schema_are_refused_without_repair(
    processed_db: Path, tmp_path: Path
) -> None:
    with closing(connect(processed_db)) as connection:
        connection.execute("CREATE TABLE email_preparations(email_id TEXT)")
        connection.commit()
    before = snapshot(processed_db)
    with pytest.raises(EmailError, match="Unversioned"):
        migrate_preparation_schema(processed_db, tmp_path / "unused.db", "synthetic")
    assert snapshot(processed_db) == before
    assert not (tmp_path / "unused.db").exists()


def test_resume_exact_dedup_preserves_every_reference_and_upstream_data(prepared_db: Path) -> None:
    with closing(connect(prepared_db, read_only=True)) as connection:
        original = connection.execute("SELECT * FROM email_items ORDER BY email_id").fetchall()
        processed = connection.execute("SELECT * FROM email_processed ORDER BY email_id").fetchall()
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        assert prepare_emails(store, options=OPTIONS, limit=1)["prepared"] == 1
        first = store.connection.execute("SELECT * FROM email_preparations").fetchall()
        assert prepare_emails(store, options=OPTIONS)["prepared"] == 4
        assert prepare_emails(store, options=OPTIONS)["attempted"] == 0
        assert (
            store.connection.execute(
                "SELECT * FROM email_preparations WHERE email_id='email-000'"
            ).fetchall()
            == first
        )
        preview = store.preview_preparation("email-000", OPTIONS)
        chunk = preview["chunks"][0]
        assert chunk["text"] == REPEATED and chunk["exact_duplicate"]
        assert {ref["email_id"] for ref in chunk["references"]} == {"email-000", "email-001"}
        assert all("start" in ref and "section_index" in ref for ref in chunk["references"])
        report = store.evaluate_preparation(OPTIONS)
        assert report["total_chunks"] == report["unique_chunks"] + 1
        assert report["unique_chunks_with_near_matches"] == 2
        assert report["excluded_characters"] > 0
        assert "PRIVATE" not in json.dumps(report) and "example.test" not in json.dumps(report)
        assert (
            store.connection.execute("SELECT * FROM email_items ORDER BY email_id").fetchall()
            == original
        )
        assert (
            store.connection.execute("SELECT * FROM email_processed ORDER BY email_id").fetchall()
            == processed
        )


def test_near_matches_stay_separate_and_preview_updates_for_later_arrivals(
    prepared_db: Path,
) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        prepare_emails(store, options=OPTIONS, limit=3)
        assert not store.preview_preparation("email-002", OPTIONS)["chunks"][0]["near_duplicates"][
            "matches"
        ]
        prepare_emails(store, options=OPTIONS)
        a = store.preview_preparation("email-002", OPTIONS)["chunks"][0]
        b = store.preview_preparation("email-003", OPTIONS)["chunks"][0]
        assert a["content_id"] != b["content_id"]
        match = a["near_duplicates"]["matches"][0]
        assert match["content_id"] == b["content_id"] and match["action"] == "retain_separate"
        assert match["references"][0]["email_id"] == "email-003"
        assert a["text"] == NEAR and b["text"] == NEAR.upper()


def test_options_and_rule_versions_keep_independent_representations(
    prepared_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        prepare_emails(store, options=OPTIONS)
        smaller = PreparationOptions(128, 10)
        assert prepare_emails(store, options=smaller)["prepared"] == 5
        assert store.evaluate_preparation(OPTIONS)["prepared"] == 5
        monkeypatch.setattr(email_preparation, "PREPARATION_VERSION", "synthetic-v2")
        assert prepare_emails(store, options=OPTIONS)["prepared"] == 5
        assert store.connection.execute("SELECT COUNT(*) FROM email_preparations").fetchone() == (
            15,
        )


def test_interruption_algorithm_failure_and_resume_checkpoint(prepared_db: Path) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        actual = email_preparation.prepare_email
        calls = 0

        def interrupted(
            record: dict[str, str], options: PreparationOptions
        ) -> email_preparation.PreparedEmail:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return actual(record, options)

        with patch.object(email_preparation, "prepare_email", side_effect=interrupted):
            assert prepare_emails(store, options=OPTIONS)["status"] == "interrupted"
        assert store.evaluate_preparation(OPTIONS)["prepared"] == 1
        with patch.object(
            email_preparation, "prepare_email", side_effect=ValueError("PRIVATE_FAILURE")
        ):
            result = prepare_emails(store, options=OPTIONS, limit=1)
        assert result["failed"] == 1 and result["status"] == "partial"
        preview = store.preview_preparation("email-001", OPTIONS)
        assert preview["error_code"] == "preparation_failed" and preview["chunks"] == []
        assert "PRIVATE_FAILURE" not in json.dumps(preview)
        assert prepare_emails(store, options=OPTIONS)["prepared"] == 4
        assert store.evaluate_preparation(OPTIONS)["failed_attempts"] == 0


def test_occurrence_failure_rolls_back_completion_content_and_references(prepared_db: Path) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        store.connection.execute(
            "CREATE TRIGGER fail_chunk BEFORE INSERT ON email_embedding_occurrences "
            "BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"
        )
        store.connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            prepare_emails(store, options=OPTIONS)
        assert store.evaluate_preparation(OPTIONS)["prepared"] == 0
        for table in (
            "email_preparations",
            "email_embedding_contents",
            "email_embedding_occurrences",
        ):
            assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (0,)
        store.connection.execute("DROP TRIGGER fail_chunk")
        store.connection.commit()
        assert prepare_emails(store, options=OPTIONS)["prepared"] == 5


def test_hard_stop_during_save_is_retryable(prepared_db: Path) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        with patch.object(store, "save_preparation", side_effect=KeyboardInterrupt):
            assert prepare_emails(store, options=OPTIONS)["status"] == "interrupted"
        assert store.evaluate_preparation(OPTIONS)["prepared"] == 0
        assert prepare_emails(store, options=OPTIONS)["prepared"] == 5


def test_interrupt_after_content_insert_rolls_back_entire_email(prepared_db: Path) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        with patch.object(store, "_near_matches", side_effect=KeyboardInterrupt):
            assert prepare_emails(store, options=OPTIONS)["status"] == "interrupted"
        assert store.connection.execute(
            "SELECT COUNT(*) FROM email_embedding_contents"
        ).fetchone() == (0,)
        assert store.evaluate_preparation(OPTIONS)["prepared"] == 0
        assert prepare_emails(store, options=OPTIONS)["prepared"] == 5


def test_failed_regeneration_preserves_prior_chunks_and_success(prepared_db: Path) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        prepare_emails(store, options=OPTIONS)
        before = store.preview_preparation("email-000", OPTIONS)
        # Simulate an upstream representation revision without touching an original.
        with store.connection:
            store.connection.execute(
                "UPDATE email_processed SET processed_text=processed_text || ' ' "
                "WHERE email_id='email-000'"
            )
        with patch.object(email_preparation, "prepare_email", side_effect=ValueError("PRIVATE")):
            assert prepare_emails(store, options=OPTIONS)["failed"] == 1
        after = store.preview_preparation("email-000", OPTIONS)
        assert after["prepared_at"] == before["prepared_at"]
        assert after["chunks"] == before["chunks"] and after["stale_input"]
        assert all(section["source_text"] is None for section in after["sections"])
        assert store.evaluate_preparation(OPTIONS)["stale_inputs"] == 1
        assert prepare_emails(store, options=OPTIONS)["prepared"] == 1
        assert not store.preview_preparation("email-000", OPTIONS)["stale_input"]


def test_input_change_between_selection_and_commit_is_refused(prepared_db: Path) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        record = next(store.preparation_candidates(OPTIONS))
        result = email_preparation.prepare_email(record, OPTIONS)
        with store.connection:
            store.connection.execute(
                "UPDATE email_processed SET processed_text='changed' " "WHERE email_id='email-000'"
            )
        with pytest.raises(EmailError, match="input changed"):
            store.save_preparation(record, OPTIONS, result)
        assert store.evaluate_preparation(OPTIONS)["prepared"] == 0


def test_read_only_preview_and_empty_evaluation_are_logically_immutable(prepared_db: Path) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        empty = store.evaluate_preparation(OPTIONS)
        assert empty["total_chunks"] == empty["unique_chunks"] == empty["excluded_characters"] == 0
        prepare_emails(store, options=OPTIONS)
    before = snapshot(prepared_db)
    with closing(EmailPreparationStore(prepared_db, "synthetic", read_only=True)) as store:
        assert store.preview_preparation("email-000", OPTIONS)["original_body"] == REPEATED
        assert store.evaluate_preparation(OPTIONS)["prepared"] == 5
        with pytest.raises(EmailError, match="No preparation"):
            store.preview_preparation("not-present", OPTIONS)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.connection.execute("DELETE FROM email_preparations")
    assert snapshot(prepared_db) == before


def test_unavailable_upstream_and_quarantined_archives_are_not_candidates(
    prepared_db: Path,
) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        with store.connection:
            store.connection.execute(
                "UPDATE email_processed SET error_code='processing_failed' "
                "WHERE email_id='email-000'"
            )
            store.connection.execute(
                "UPDATE email_items SET status='skipped' WHERE email_id='email-001'"
            )
        assert prepare_emails(store, options=OPTIONS)["prepared"] == 3
        with store.connection:
            store.connection.execute("UPDATE email_archives SET integrity_status='source_changed'")
        assert prepare_emails(store, options=PreparationOptions(128, 0))["attempted"] == 0


def test_near_candidate_limits_are_visible_not_silent(
    prepared_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with closing(EmailPreparationStore(prepared_db, "synthetic")) as store:
        prepare_emails(store, options=OPTIONS)
        monkeypatch.setattr(email_preparation_store, "NEAR_CANDIDATE_LIMIT", 0)
        preview = store.preview_preparation("email-002", OPTIONS)
        assert preview["chunks"][0]["near_duplicates"]["candidates_truncated"]
        assert store.evaluate_preparation(OPTIONS)["near_candidate_truncations"] == 2


def test_cli_isolation_no_services_and_content_free_statistics(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    databases = {}
    for profile in ("mom", "personal"):
        path = Path(load_config(profile)["db_path"])
        seed_processed(path, profile, [f"PRIVATE_{profile}: AB-123 at 123 €"])
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
        patch("jarvis.ingest.email.load_parser", side_effect=AssertionError("No PST")),
    ):
        assert command("prepare")[0] == 1
        assert command("preparation-migrate", "--backup", str(profile_repo / "backup.db"))[0] == 0
        assert command("prepare", "--max-chars", "128", "--overlap", "64")[0] == 1
        code, output = command("prepare", "--limit", "1")
        assert code == 0 and '"prepared": 1' in output and "PRIVATE" not in output
        assert command("prepare")[0] == 0
        code, output = command("evaluate-chunks")
        assert code == 0 and '"unique_chunks": 1' in output and "PRIVATE" not in output
        code, output = command("preview-chunks", "email-000")
        assert code == 0 and "PRIVATE_mom" in output and "PRIVATE_personal" not in output
        assert command("preview-chunks", "missing")[0] == 1
        with patch.object(email_preparation, "prepare_email", side_effect=KeyboardInterrupt):
            assert command("prepare", "--max-chars", "128", "--overlap", "0")[0] == 130
        with patch.object(
            email_preparation, "prepare_email", side_effect=ValueError("PRIVATE_ERROR")
        ):
            code, output = command("prepare", "--max-chars", "128", "--overlap", "0")
            assert code == 1 and "PRIVATE_ERROR" not in output
    assert snapshot(databases["personal"]) == personal_before


def test_missing_database_is_not_initialized(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"
    with pytest.raises(Exception):
        EmailPreparationStore(path, "synthetic")
    assert not path.exists()
