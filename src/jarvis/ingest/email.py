"""Streaming, resumable PST imports; original text remains separate from future AI data."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, closing
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from jarvis.email_store import EmailError, EmailStore, connect, email_objects, json_value
from jarvis.pst import (
    BODY_PROPERTIES,
    attempt,
    decode_text,
    email_kind,
    header_fields,
    html_to_text,
    metadata,
    recipient_metadata,
)
from jarvis.store import SummaryStore


class SourceChanged(EmailError):
    pass


class ArchiveFailure(EmailError):
    pass


def load_parser() -> Any:
    try:
        return importlib.import_module("pypff")
    except ImportError:
        raise EmailError("Install the optional PST dependency in the email environment") from None


@dataclass
class FolderEvent:
    native: Any
    locator: tuple[int, ...]
    names: list[str | None]
    count: int
    error: str | None = None


def walk_folders(root: Any) -> Iterator[FolderEvent]:
    seen = set()
    pending = [(iter([(root, ())]), [])]
    while pending:
        children, parent_names = pending[-1]
        child = next(children, None)
        if child is None:
            pending.pop()
            continue
        native, locator = child
        if isinstance(native, Exception):
            yield FolderEvent(None, locator, [*parent_names, None], 0, "folder_read")
            continue
        identifier = attempt(lambda: native.identifier)["value"]
        if identifier is not None and identifier in seen:
            yield FolderEvent(None, locator, [*parent_names, None], 0, "folder_cycle")
            continue
        if identifier is not None:
            seen.add(identifier)
        name = attempt(lambda: native.name)
        names = [*parent_names, name["value"] if isinstance(name["value"], str) else None]
        messages = attempt(lambda: native.number_of_sub_messages)
        subfolders = attempt(lambda: native.number_of_sub_folders)
        count = messages["value"]
        count_valid = isinstance(count, int) and count >= 0
        folder_count = subfolders["value"]
        children_valid = isinstance(folder_count, int) and folder_count >= 0
        error = (
            None
            if count_valid and children_valid and name["status"] != "error"
            else "folder_metadata"
        )
        yield FolderEvent(native, locator, names, count if count_valid else 0, error)
        if len(locator) >= 4096:
            yield FolderEvent(None, (*locator, -1), names, 0, "folder_depth")
        elif children_valid:
            pending.append((folder_children(native, locator, folder_count), names))


def folder_children(
    parent: Any, locator: tuple[int, ...], count: int
) -> Iterator[tuple[Any, tuple[int, ...]]]:
    for index in range(count):
        try:
            child = parent.get_sub_folder(index)
        except Exception as error:
            child = error
        yield child, (*locator, index)


def item_identity(
    archive_id: str, kind: str, locator: tuple[int, ...], index: int | None
) -> tuple[str, str]:
    key = json_value([kind, locator, index])
    return hashlib.sha256((archive_id + ":" + key).encode()).hexdigest(), key


def empty_item(
    archive_id: str, kind: str, folder: FolderEvent, index: int | None
) -> dict[str, Any]:
    email_id, key = item_identity(archive_id, kind, folder.locator, index)
    return {
        "email_id": email_id,
        "archive_id": archive_id,
        "item_key": key,
        "item_kind": kind,
        "status": "failed",
        "folder_path_json": json_value(folder.names),
        "folder_locator_json": json_value(folder.locator),
        "item_index": index,
        "pst_identifier": None,
        "original_body": None,
        "body_source": None,
        "metadata_json": "{}",
        "body_json": "{}",
        "headers_json": "{}",
        "recipients_json": "{}",
        "warnings_json": "[]",
        "error_code": None,
    }


def original_body(message: Any, values: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    evidence: dict[str, Any] = {"source": "none", "availability": {}, "text_status": "absent"}
    codepage = values["internet_codepage"]["value"] or values["message_codepage"]["value"]
    empty = False
    for name, attribute in (("plain", "plain_text_body"), ("html", "html_body")):
        raw = attempt(lambda: getattr(message, attribute))
        evidence["availability"][name] = {"status": raw["status"]}
        if raw["status"] not in ("present", "empty"):
            continue
        try:
            text, codec = decode_text(raw["value"], codepage)
            if name == "html":
                text = html_to_text(text)
            evidence["availability"][name]["character_length"] = len(text)
            if text:
                evidence.update(
                    source=name, decoding=codec, character_length=len(text), text_status="present"
                )
                return text, evidence
            empty = True
        except Exception:
            evidence["availability"][name] = {"status": "error"}
    statuses = [state["status"] for state in evidence["availability"].values()]
    evidence["text_status"] = "error" if "error" in statuses else "empty" if empty else "absent"
    return "" if empty else None, evidence


def warnings_in(value: Any, prefix: str) -> list[str]:
    result = []
    if isinstance(value, dict):
        if value.get("status") in ("error", "unsupported"):
            result.append(f"{prefix}:{value['status']}")
        else:
            for name, child in value.items():
                if name != "value" or isinstance(child, list):
                    result.extend(warnings_in(child, f"{prefix}.{name}"))
    elif isinstance(value, list):
        for child in value:
            result.extend(warnings_in(child, prefix))
    return result


def extract_message(message: Any, item: dict[str, Any]) -> dict[str, Any]:
    values, record = metadata(message)
    kind = email_kind(values)
    flags = values["message_flags"]["value"]
    direction = (
        "draft"
        if isinstance(flags, int) and flags & 8
        else "sent" if isinstance(flags, int) and flags & 32 else "unknown"
    )
    bodies = {name: record.presence(tag) for name, tag in BODY_PROPERTIES.items()}
    item["metadata_json"] = json_value(
        {
            "fields": values,
            "body_properties": bodies,
            "direction": direction,
            "record_status": record.status,
        }
    )
    identifier = values["item_identifier"]["value"]
    item["pst_identifier"] = str(identifier) if identifier is not None else None
    if kind in ("non_email", "unsupported"):
        item.update(
            status="skipped", error_code="non_email" if kind == "non_email" else "unsupported_class"
        )
        return item
    if kind == "unknown":
        item["error_code"] = (
            "message_class_unreadable" if record.status == "error" else "message_class_missing"
        )
        return item
    recipients = recipient_metadata(message)
    headers = header_fields(values)
    text, body = original_body(message, values)
    warnings = warnings_in(values, "metadata") + warnings_in(recipients, "recipients")
    warnings += warnings_in(headers, "headers") + warnings_in(body["availability"], "body")
    if record.status != "present":
        warnings.append(f"property_table:{record.status}")
    if body.get("decoding") == "utf-8-replacement":
        warnings.append("body:replacement_decoding")
    item.update(
        original_body=text,
        body_source=body["source"],
        body_json=json_value(body),
        headers_json=json_value(headers),
        recipients_json=json_value(recipients),
        warnings_json=json_value(warnings),
        status="imported",
    )
    if body["text_status"] == "error":
        item.update(status="failed", error_code="body_read")
    elif body["source"] == "none" and bodies["rtf"]["status"] == "present":
        item.update(status="failed", error_code="rtf_only_unsupported")
    return item


def hash_source(stream: BinaryIO, progress: Callable[[str], None]) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
        if size % (256 * 1024 * 1024) == 0:
            progress(f"Hashed {size // (1024 * 1024)} MiB.")
    stream.seek(0)
    return digest.hexdigest()


def stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino


def assert_source_unchanged(source: Path, stream: BinaryIO, before: os.stat_result) -> None:
    try:
        matches = stat_identity(source.stat()) == stat_identity(before)
        matches = matches and stat_identity(os.fstat(stream.fileno())) == stat_identity(before)
    except OSError:
        matches = False
    if not matches:
        raise SourceChanged("PST changed during import; archive requires review")


def optional_read_store(db_path: Path, profile: str) -> EmailStore | None:
    if not db_path.exists():
        return None
    SummaryStore(str(db_path), read_only=True)
    with closing(connect(db_path, read_only=True)) as connection:
        installed = connection.execute(
            "SELECT value FROM _jarvis_meta WHERE key='email_schema_version'"
        ).fetchone()
        objects = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        objects &= set(email_objects())
    return EmailStore(db_path, profile, read_only=True) if installed or objects else None


def import_pst(
    source: Path,
    db_path: Path,
    profile: str,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    parser_module: Any = None,
    progress: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise EmailError("Limit must be a positive integer")
    source = source.expanduser().resolve(strict=True)
    db_path = db_path.resolve()
    if not source.is_file() or source.suffix.lower() != ".pst":
        raise EmailError("Source must be a regular PST file")
    if source == db_path or any(
        source == Path(f"{db_path}{suffix}") for suffix in ("-wal", "-shm", "-journal")
    ):
        raise EmailError("Source overlaps profile database storage")
    if parser_module is None:
        parser_module = load_parser()
    counts = {
        name: 0
        for name in (
            "attempted",
            "imported",
            "would_import",
            "skipped",
            "failed",
            "previously_imported",
            "previously_skipped",
            "warning_messages",
        )
    }
    run_id = str(uuid4())
    status = "running"
    error_code = None
    with ExitStack() as stack:
        store = optional_read_store(db_path, profile) if dry_run else EmailStore(db_path, profile)
        if store:
            stack.callback(store.close)
        stream = stack.enter_context(source.open("rb"))
        before = os.fstat(stream.fileno())
        progress("Hashing the read-only source for archive identity.")
        archive_id = hash_source(stream, progress)
        assert_source_unchanged(source, stream, before)
        version = str(attempt(lambda: parser_module.get_version())["value"] or "unknown")
        if store:
            store.require_archive_compatible(archive_id, version)
        if store and not dry_run:
            store.start_run(archive_id, before.st_size, version, source, run_id, counts)
        archive = None
        opened = False

        def persist(item: dict[str, Any], *, check_source: bool = True) -> None:
            nonlocal counts
            if check_source:
                assert_source_unchanged(source, stream, before)
            updated = dict(counts)
            updated["attempted"] += 1
            outcome = item["status"]
            updated["would_import" if dry_run and outcome == "imported" else outcome] += 1
            if json_value([]) != item["warnings_json"]:
                updated["warning_messages"] += 1
            if store and not dry_run:
                store.record(run_id, item, updated)
            counts = updated

        try:
            progress("Opening PST read-only; streaming folder items.")
            diagnostic = empty_item(archive_id, "archive", FolderEvent(None, (), [], 0), None)
            try:
                archive = parser_module.file()
                archive.open_file_object(stream, mode="r")
                opened = True
                root = archive.get_root_folder()
                if root is None:
                    raise ArchiveFailure("PST root folder is unavailable")
            except Exception:
                diagnostic["error_code"] = "archive_open" if not opened else "root_folder"
                persist(diagnostic)
                raise ArchiveFailure("PST cannot be enumerated") from None
            if store and not dry_run:
                store.resolve_failure(diagnostic["email_id"])
            stop = False
            for folder in walk_folders(root):
                diagnostic = empty_item(archive_id, "folder", folder, None)
                if folder.error:
                    diagnostic["error_code"] = folder.error
                    persist(diagnostic)
                elif store and not dry_run:
                    store.resolve_failure(diagnostic["email_id"])
                if limit is not None and counts["attempted"] >= limit:
                    stop = True
                    break
                for index in range(folder.count):
                    item = empty_item(archive_id, "message", folder, index)
                    previous = store.item_status(item["email_id"]) if store else None
                    if previous in ("imported", "skipped"):
                        counts[
                            (
                                "previously_imported"
                                if previous == "imported"
                                else "previously_skipped"
                            )
                        ] += 1
                        if store and not dry_run:
                            with store.connection:
                                store.save_counts(run_id, counts)
                        previous_count = (
                            counts["previously_imported"] + counts["previously_skipped"]
                        )
                        if previous_count % 250 == 0:
                            progress(f"Found {previous_count} previously recorded items.")
                        continue
                    try:
                        message = folder.native.get_sub_message(index)
                        item = extract_message(message, item)
                    except Exception:
                        item.update(status="failed", error_code="message_read")
                    persist(item)
                    if counts["attempted"] % 25 == 0:
                        progress(
                            f"Attempted {counts['attempted']}; imported {counts['imported']}; "
                            f"failed {counts['failed']}."
                        )
                    if limit is not None and counts["attempted"] >= limit:
                        stop = True
                        break
                if stop:
                    break
            status = "limited" if stop else "complete"
        except KeyboardInterrupt:
            status, error_code = "interrupted", "keyboard_interrupt"
            if store and not dry_run:
                row = store.connection.execute(
                    "SELECT counts_json FROM email_import_runs WHERE run_id=?", (run_id,)
                ).fetchone()
                counts = json.loads(row[0])
        except SourceChanged:
            status, error_code = "source_changed", "source_changed"
        except ArchiveFailure:
            status, error_code = "failed", "archive_read"
        except Exception:
            status, error_code = "failed", "archive_or_storage_failure"
            if store and not dry_run:
                row = store.connection.execute(
                    "SELECT counts_json FROM email_import_runs WHERE run_id=?", (run_id,)
                ).fetchone()
                counts = json.loads(row[0])
        finally:
            if opened:
                try:
                    archive.close()
                except KeyboardInterrupt:
                    status, error_code = "interrupted", "keyboard_interrupt"
                except Exception:
                    item = empty_item(archive_id, "archive", FolderEvent(None, (), [], 0), None)
                    item["error_code"] = "archive_close"
                    persist(item, check_source=False)
                    error_code = "archive_close"
                    if status not in ("interrupted", "source_changed"):
                        status = "failed"
            try:
                assert_source_unchanged(source, stream, before)
                if status in ("complete", "limited"):
                    progress("Verifying the source hash after import.")
                    if hash_source(stream, progress) != archive_id:
                        raise SourceChanged("PST content changed")
            except KeyboardInterrupt:
                status, error_code = "interrupted", "keyboard_interrupt"
            except SourceChanged:
                status, error_code = "source_changed", "source_changed"
            except Exception:
                status, error_code = "source_changed", "source_verification_failed"
            if counts["failed"] and status in ("complete", "limited"):
                status = "partial"
            if store and not dry_run:
                store.finish_run(run_id, status, counts, error_code)
        return {
            "run_id": run_id if not dry_run else None,
            "archive_id": archive_id,
            "status": status,
            "dry_run": dry_run,
            "migration_required": store is None,
            "counts": counts,
            "error_code": error_code,
        }
