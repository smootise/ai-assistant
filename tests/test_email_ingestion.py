"""Synthetic PST imports into profile-isolated temporary SQLite databases."""

from contextlib import closing
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from jarvis import cli
from jarvis.email_store import EmailError, EmailStore, migrate_email_schema
from jarvis.ingest import email
from jarvis.store import SummaryStore
from tests.test_pst_discovery import Archive, Folder, Message


@pytest.fixture
def email_import(tmp_path: Path) -> Any:
    db = tmp_path / "mom.db"
    SummaryStore(str(db))
    migrate_email_schema(db, tmp_path / "before-email.db", "mom")
    source = tmp_path / "synthetic.pst"
    source.write_bytes(b"synthetic immutable PST")

    def run(messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        archive = Archive(Folder("Inbox", messages, kwargs.pop("children", [])))
        module = SimpleNamespace(file=lambda: archive, get_version=lambda: "synthetic-v1")
        before = source.read_bytes(), source.stat().st_mtime_ns
        result = email.import_pst(source, db, "mom", parser_module=module, **kwargs)
        assert before == (source.read_bytes(), source.stat().st_mtime_ns)
        assert archive.closed
        return result

    return SimpleNamespace(run=run, db=db, source=source)


def items(db: Path, *, failures: bool = False) -> list[dict[str, Any]]:
    with closing(EmailStore(db, "mom", read_only=True)) as store:
        return [store.get_email(row["email_id"]) for row in store.list_items(100, failures)]


def test_original_text_metadata_flags_and_no_attachment_extraction(email_import: Any) -> None:
    original = "  Original text\r\n\r\n-- Signature\r\n> quoted reply\nFwd: previous message  "
    message = Message(
        1,
        plain=original,
        attachment=True,
        values={
            "message_flags": 8,
            "conversation_index": b"index",
            "conversation_topic": "topic",
            "in_reply_to": "<parent>",
            "references": "<root> <parent>",
            "transport_headers": "Message-ID: <transport>\nReferences: <root>\n\n",
        },
    )
    message.get_attachment = Mock(
        side_effect=AssertionError("No attachment objects during ingestion")
    )
    result = email_import.run([message])
    assert result["counts"]["imported"] == 1
    row = items(email_import.db)[0]
    assert row["original_body"] == original
    assert row["body_source"] == "plain"
    assert row["metadata"]["direction"] == "draft"
    assert row["metadata"]["fields"]["attachment_count"]["value"] == 1
    assert row["metadata"]["fields"]["conversation_index"]["value"]["data"] == b"index".hex()
    assert row["headers"]["references"]["value"] == ["<root>"]
    assert row["folder_path"] == ["Inbox"]
    assert row["pst_identifier"] == "10001"
    assert row["recipients"]["value"][0]["address"]["value"] == "recipient@example.invalid"
    message.get_attachment.assert_not_called()


def test_missing_fields_and_received_timestamp_never_imply_received(email_import: Any) -> None:
    result = email_import.run(
        [
            Message(
                values={
                    "internet_message_id": None,
                    "subject": "",
                    "sender_name": None,
                    "message_flags": 0,
                }
            )
        ]
    )
    assert result["status"] == "complete"
    row = items(email_import.db)[0]
    assert row["metadata"]["fields"]["internet_message_id"]["status"] == "absent"
    assert row["metadata"]["fields"]["subject"]["status"] == "empty"
    assert row["metadata"]["direction"] == "unknown"


def test_limit_resume_and_duplicate_message_ids_preserve_distinct_items(email_import: Any) -> None:
    messages = [Message(index, values={"internet_message_id": "<same>"}) for index in range(5)]
    first = email_import.run(messages, limit=2)
    assert first["status"] == "limited"
    assert first["counts"]["imported"] == 2
    second = email_import.run(messages, limit=2)
    assert second["counts"]["previously_imported"] == 2
    assert second["counts"]["imported"] == 2
    final = email_import.run(messages)
    assert final["counts"]["previously_imported"] == 4
    assert final["counts"]["imported"] == 1
    repeated = email_import.run(messages)
    assert repeated["counts"]["previously_imported"] == 5
    assert repeated["counts"]["attempted"] == 0
    assert len(items(email_import.db)) == 5


def test_missing_identifiers_use_stable_archive_scoped_locations(email_import: Any) -> None:
    messages = [Message(index, values={"internet_message_id": None}) for index in range(3)]
    for message in messages:
        message.identifier = None
    email_import.run(messages, children=[Folder("Other", [Message(9)])])
    ids = [row["email_id"] for row in items(email_import.db)]
    email_import.run(messages, children=[Folder("Other", [Message(9)])])
    assert ids == [row["email_id"] for row in items(email_import.db)]
    assert len(ids) == 4


def test_identical_archive_copy_resumes_but_different_bytes_do_not_deduplicate(
    email_import: Any, tmp_path: Path
) -> None:
    message = Message()
    email_import.run([message])
    copied = tmp_path / "copied.pst"
    copied.write_bytes(email_import.source.read_bytes())
    module = SimpleNamespace(
        file=lambda: Archive(Folder("Inbox", [message])), get_version=lambda: "synthetic-v1"
    )
    resumed = email.import_pst(copied, email_import.db, "mom", parser_module=module)
    assert resumed["counts"]["previously_imported"] == 1
    copied.write_bytes(b"different archive, same Message-ID")
    distinct = email.import_pst(copied, email_import.db, "mom", parser_module=module)
    assert distinct["counts"]["imported"] == 1
    assert len(items(email_import.db)) == 2


def test_interruption_commits_progress_and_resumes(email_import: Any) -> None:
    messages = [Message(), KeyboardInterrupt(), Message(2)]
    interrupted = email_import.run(messages)
    assert interrupted["status"] == "interrupted"
    assert interrupted["counts"]["imported"] == 1
    assert len(items(email_import.db)) == 1
    resumed = email_import.run([Message(), Message(1), Message(2)])
    assert resumed["counts"]["previously_imported"] == 1
    assert resumed["counts"]["imported"] == 2


def test_failed_items_are_recorded_and_retried(email_import: Any) -> None:
    result = email_import.run(
        [
            OSError("PRIVATE_ERROR"),
            Message(1, plain=OSError("PRIVATE"), html=OSError("PRIVATE")),
            Message(2),
        ]
    )
    assert result["status"] == "partial"
    assert result["counts"]["failed"] == 2
    failed = items(email_import.db, failures=True)
    assert {item["error_code"] for item in failed} == {"message_read", "body_read"}
    assert "PRIVATE_ERROR" not in json.dumps(failed)
    resumed = email_import.run([Message(), Message(1), Message(2)])
    assert resumed["counts"]["imported"] == 2
    assert resumed["counts"]["previously_imported"] == 1
    assert items(email_import.db, failures=True) == []


def test_unreadable_folder_is_recorded_then_resolved(email_import: Any) -> None:
    result = email_import.run([Message()], children=[OSError("PRIVATE_FOLDER")])
    assert result["counts"]["failed"] == 1
    assert items(email_import.db, failures=True)[0]["error_code"] == "folder_read"
    result = email_import.run([Message()], children=[Folder("Recovered", [Message(1)])])
    assert result["counts"]["imported"] == 1
    assert items(email_import.db, failures=True) == []


def test_non_email_and_missing_class_are_explicit(email_import: Any) -> None:
    result = email_import.run(
        [
            Message(values={"message_class": "IPM.Appointment"}),
            Message(1, values={"message_class": None}),
            Message(2, plain=None, rtf=True),
        ]
    )
    assert result["counts"]["skipped"] == 1
    assert result["counts"]["failed"] == 2
    assert {row["error_code"] for row in items(email_import.db, failures=True)} == {
        "message_class_missing",
        "rtf_only_unsupported",
    }


def test_html_fallback_and_encoding_are_preserved(email_import: Any) -> None:
    email_import.run(
        [
            Message(
                plain=None, html=b"<p>caf\xe9</p><p>quoted</p>", values={"internet_codepage": 1252}
            )
        ]
    )
    row = items(email_import.db)[0]
    assert row["original_body"] == "caf\u00e9\nquoted"
    assert row["body_source"] == "html"
    assert row["body"]["decoding"] == "cp1252"
    assert row["body"]["availability"]["plain"]["status"] == "absent"


def test_dry_run_has_no_sqlite_changes_and_counts_pending(email_import: Any) -> None:
    before = email_import.db.read_bytes(), email_import.db.stat().st_mtime_ns
    result = email_import.run([Message(), Message(1)], dry_run=True, limit=1)
    assert result["counts"]["imported"] == 0
    assert result["counts"]["would_import"] == 1
    assert before == (email_import.db.read_bytes(), email_import.db.stat().st_mtime_ns)
    assert items(email_import.db) == []


def test_dry_run_before_initialization_creates_nothing(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.pst"
    source.write_bytes(b"synthetic")
    db = tmp_path / "missing" / "profile.db"
    module = SimpleNamespace(
        file=lambda: Archive(Folder("Inbox", [Message()])), get_version=lambda: "test"
    )
    result = email.import_pst(source, db, "mom", dry_run=True, parser_module=module)
    assert result["migration_required"]
    assert result["counts"]["would_import"] == 1
    assert not db.parent.exists()
    with pytest.raises(Exception):
        email.import_pst(source, db, "mom", parser_module=module)
    assert not db.parent.exists()


def test_optional_metadata_errors_are_visible_warnings(email_import: Any) -> None:
    result = email_import.run([Message(values={"sender_name": OSError("SECRET_NAME")})])
    assert result["counts"]["imported"] == 1
    assert result["counts"]["warning_messages"] == 1
    row = items(email_import.db)[0]
    assert "metadata.sender_name:error" in row["warnings"]
    assert "SECRET_NAME" not in json.dumps(row)


def test_cli_profile_isolation_and_private_show(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from jarvis.config import load_config

    personal = load_config("personal")
    mom = load_config("mom")
    personal_store = SummaryStore(personal["db_path"])
    personal_store.insert_source_file(
        {
            "source_file_id": "preserve",
            "source_kind": "chatgpt",
            "original_filename": "synthetic",
            "storage_path": "synthetic",
            "sha256": "synthetic",
        }
    )
    before = personal_store.count_all()

    def command(*arguments: str) -> int:
        monkeypatch.setattr(sys, "argv", ["jarvis", "--profile", "mom", *arguments])
        return cli.main()

    assert command("email", "init") == 0
    assert command("email", "migrate", "--backup", str(profile_repo / "backup.db")) == 0
    source = profile_repo / "synthetic.pst"
    source.write_bytes(b"synthetic")
    module = SimpleNamespace(
        file=lambda: Archive(Folder("Inbox", [Message()])), get_version=lambda: "test"
    )
    with patch.object(email, "load_parser", return_value=module):
        assert command("ingest", "pst", "--pst", str(source), "--limit", "100") == 0
    assert command("email", "status") == 0
    assert command("email", "list") == 0
    assert "PRIVATE_BODY_SENTINEL" not in capsys.readouterr().out
    with closing(EmailStore(Path(mom["db_path"]), "mom", read_only=True)) as store:
        email_id = store.list_items()[0]["email_id"]
    assert command("email", "show", email_id) == 0
    assert "PRIVATE_BODY_SENTINEL" in capsys.readouterr().out
    assert personal_store.count_all() == before
    assert len(items(Path(mom["db_path"]))) == 1


def test_cli_refuses_cross_profile_source(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = profile_repo / "inbox" / "synthetic.pst"
    source.parent.mkdir()
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(
        sys,
        "argv",
        ["jarvis", "--profile", "mom", "ingest", "pst", "--pst", str(source), "--dry-run"],
    )
    assert cli.main() == 1
    assert not (profile_repo / "data").exists()


def test_dry_run_after_partial_import_counts_only_remaining(email_import: Any) -> None:
    messages = [Message(), Message(1)]
    email_import.run(messages, limit=1)
    result = email_import.run(messages, dry_run=True)
    assert result["counts"]["previously_imported"] == 1
    assert result["counts"]["would_import"] == 1
    assert len(items(email_import.db)) == 1


def test_no_sample_length_cap_on_original_text(email_import: Any) -> None:
    text = "untrimmed original\n" * 20000
    email_import.run([Message(plain=text)])
    assert items(email_import.db)[0]["original_body"] == text


def test_atomic_success_guard_preserves_original_even_after_stale_status_read(
    email_import: Any,
) -> None:
    email_import.run([Message(plain="preserve this original")])
    with patch.object(EmailStore, "item_status", return_value=None):
        result = email_import.run([Message(plain="must not overwrite")])
    assert result["status"] == "failed"
    assert items(email_import.db)[0]["original_body"] == "preserve this original"


def test_root_and_close_failures_are_recorded(email_import: Any) -> None:
    archive = Archive(Folder("Inbox", [Message()]))
    archive.get_root_folder = Mock(side_effect=OSError("PRIVATE_ROOT"))
    module = SimpleNamespace(file=lambda: archive, get_version=lambda: "test")
    result = email.import_pst(email_import.source, email_import.db, "mom", parser_module=module)
    assert result["status"] == "failed"
    assert items(email_import.db, failures=True)[0]["error_code"] == "root_folder"
    archive.get_root_folder = Mock(return_value=Folder("Inbox", [Message()]))
    archive.close = Mock(side_effect=OSError("PRIVATE_CLOSE"))
    result = email.import_pst(email_import.source, email_import.db, "mom", parser_module=module)
    assert result["status"] == "failed"
    assert items(email_import.db, failures=True)[0]["error_code"] == "archive_close"
    assert "PRIVATE_ROOT" not in json.dumps(result)


def test_parser_open_failure_and_recovery(email_import: Any) -> None:
    archive = Archive(Folder("Inbox", [Message()]))
    archive.open_file_object = Mock(side_effect=OSError("PRIVATE_ARCHIVE"))
    module = SimpleNamespace(file=lambda: archive, get_version=lambda: "synthetic-v1")
    result = email.import_pst(email_import.source, email_import.db, "mom", parser_module=module)
    assert result["status"] == "failed"
    assert result["counts"]["failed"] == 1
    assert items(email_import.db, failures=True)[0]["error_code"] == "archive_open"
    result = email_import.run([Message()])
    assert result["counts"]["imported"] == 1
    assert items(email_import.db, failures=True) == []


def test_record_and_counts_rollback_together(email_import: Any) -> None:
    original = EmailStore.save_counts

    def fail_counts(store: EmailStore, run_id: str, counts: dict[str, int]) -> None:
        if counts["imported"]:
            raise RuntimeError("synthetic storage failure")
        original(store, run_id, counts)

    with patch.object(EmailStore, "save_counts", fail_counts):
        result = email_import.run([Message()])
    assert result["status"] == "failed"
    assert result["counts"]["imported"] == 0
    assert items(email_import.db) == []
    assert email_import.run([Message()])["counts"]["imported"] == 1


def test_interruption_after_commit_reports_committed_counts(email_import: Any) -> None:
    original = EmailStore.record

    def interrupted_record(
        store: EmailStore, run_id: str, item: dict[str, Any], counts: dict[str, int]
    ) -> None:
        original(store, run_id, item, counts)
        raise KeyboardInterrupt()

    with patch.object(EmailStore, "record", interrupted_record):
        result = email_import.run([Message()])
    assert result["status"] == "interrupted"
    assert result["counts"]["imported"] == 1
    assert len(items(email_import.db)) == 1


def test_source_change_quarantines_archive_without_overwriting_originals(email_import: Any) -> None:
    class ChangingMessage(Message):
        @property
        def plain_text_body(self) -> str:
            email_import.source.write_bytes(b"changed synthetic source")
            return "changed source body"

    archive = Archive(Folder("Inbox", [Message(), ChangingMessage(1)]))
    module = SimpleNamespace(file=lambda: archive, get_version=lambda: "test")
    result = email.import_pst(email_import.source, email_import.db, "mom", parser_module=module)
    assert result["status"] == "source_changed"
    assert result["counts"]["imported"] == 1
    row = items(email_import.db)[0]
    assert row["original_body"] == "PRIVATE_BODY_SENTINEL"
    assert row["archive_integrity_status"] == "source_changed"
    email_import.source.write_bytes(b"synthetic immutable PST")
    with pytest.raises(EmailError, match="quarantined"):
        email.import_pst(email_import.source, email_import.db, "mom", parser_module=module)


def test_final_hash_detects_change_even_if_size_and_mtime_are_restored(email_import: Any) -> None:
    original_stat = email_import.source.stat()

    class ChangingMessage(Message):
        @property
        def plain_text_body(self) -> str:
            original = email_import.source.read_bytes()
            email_import.source.write_bytes(b"X" * len(original))
            os.utime(email_import.source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            return "synthetic text"

    module = SimpleNamespace(
        file=lambda: Archive(Folder("Inbox", [ChangingMessage()])), get_version=lambda: "test"
    )
    result = email.import_pst(email_import.source, email_import.db, "mom", parser_module=module)
    assert result["status"] == "source_changed"
    assert items(email_import.db)[0]["archive_integrity_status"] == "source_changed"


def test_parser_version_change_is_refused_for_resume_and_dry_run(email_import: Any) -> None:
    email_import.run([Message()])
    module = SimpleNamespace(
        file=lambda: Archive(Folder("Inbox", [Message()])), get_version=lambda: "different"
    )
    for dry_run in (False, True):
        with pytest.raises(EmailError, match="parser version"):
            email.import_pst(
                email_import.source, email_import.db, "mom", dry_run=dry_run, parser_module=module
            )


def test_non_email_is_not_read_again(email_import: Any) -> None:
    email_import.run([Message(values={"message_class": "IPM.Appointment"})])
    result = email_import.run([OSError("must not be read")])
    assert result["counts"]["previously_skipped"] == 1
    assert result["counts"]["failed"] == 0


def test_cli_import_has_no_inference_or_vector_client(
    profile_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis.config import load_config

    config = load_config("mom")
    SummaryStore(config["db_path"])
    migrate_email_schema(Path(config["db_path"]), profile_repo / "backup.db", "mom")
    source = profile_repo / "synthetic.pst"
    source.write_bytes(b"synthetic")
    module = SimpleNamespace(
        file=lambda: Archive(Folder("Inbox", [Message()])), get_version=lambda: "test"
    )
    monkeypatch.setattr(
        sys, "argv", ["jarvis", "--profile", "mom", "ingest", "pst", "--pst", str(source)]
    )
    with (
        patch.object(email, "load_parser", return_value=module),
        patch("jarvis.cli._build_memory_layer", side_effect=AssertionError("No memory layer")),
        patch("jarvis.clients.OllamaClient", side_effect=AssertionError("No inference")),
        patch("jarvis.clients.VectorStore", side_effect=AssertionError("No vectors")),
        patch("jarvis.clients.EmbeddingClient", side_effect=AssertionError("No embeddings")),
    ):
        assert cli.main() == 0
