"""Incremental PST inventory and bounded, deterministic diversity sampling."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Iterator

from tools.pst_parser import (
    decode_text,
    html_to_text,
    header_fields,
    email_kind,
    ATTACHMENT_PROPERTIES,
    BODY_PROPERTIES,
    RECIPIENT_PROPERTIES,
    attachment_metadata,
    attempt,
    field,
    metadata,
    recipient_metadata,
    serialize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    "pst_inventory.json",
    "pst_inventory.md",
    "pst_samples_private.json",
    "pst_samples_structure.json",
)
STATES = ("present", "empty", "absent", "unsupported", "error")


def body_format(presence: dict[str, dict[str, Any]]) -> str:
    if any(item["status"] in ("error", "unsupported") for item in presence.values()):
        return "unknown"
    if presence["plain"]["status"] == "present":
        return "plain"
    if presence["html"]["status"] == "present":
        return "html_only"
    if presence["rtf"]["status"] == "present":
        return "rtf_only"
    return "none"


def sample_body(message: Any, values: dict[str, Any], limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {"availability": {}, "text": field(), "source": "none"}
    codepage = values["internet_codepage"]["value"] or values["message_codepage"]["value"]
    for name, attr in (("plain", "plain_text_body"), ("html", "html_body")):
        raw = attempt(lambda: getattr(message, attr))
        result["availability"][name] = {"status": raw["status"]}
        if raw["status"] not in ("present", "empty"):
            continue
        try:
            text, codec = decode_text(raw["value"], codepage)
            if name == "html":
                text = html_to_text(text)
            result["availability"][name]["character_length"] = len(text)
            if text:
                result.update(
                    text=field(text[:limit]),
                    source=name,
                    character_length=len(text),
                    truncated=len(text) > limit,
                    decoding=codec,
                )
                break
            result["text"] = field("")
        except Exception:
            result["availability"][name] = {"status": "error"}
    # RTF is deliberately not fetched or decompressed.
    if result["source"] == "none" and result["text"]["status"] != "empty":
        statuses = [v["status"] for v in result["availability"].values()]
        status = (
            "error"
            if "error" in statuses
            else "unsupported" if "unsupported" in statuses else "absent"
        )
        result["text"] = field(status=status)
    return result


def direction(values: dict[str, Any]) -> str:
    flags = values["message_flags"]["value"]
    if isinstance(flags, int):
        if flags & 0x08:
            return "draft"
        if flags & 0x20:
            return "sent"
    # Delivery time is evidence of delivery, not proof of the archive owner's identity.
    if values["received_time"]["status"] == "present":
        return "received_evidence"
    return "unknown"


@dataclass(frozen=True)
class Candidate:
    folder: int
    index: int
    priority: float
    features: frozenset[tuple[str, Any]]


class Sampler:
    def __init__(self, count: int, seed: int) -> None:
        self.count = count
        self.rng = random.Random(seed)
        self.uniform: list[Candidate] = []
        self.strata: dict[tuple[str, Any], Candidate] = {}
        self.seen = 0

    def add(self, folder: int, index: int, values: dict[str, Any], format_name: str) -> None:
        self.seen += 1
        if not self.count:
            return
        date = values["sent_time"]["value"] or values["received_time"]["value"]
        subject = values["subject"]["value"]
        subject = subject.casefold().lstrip() if isinstance(subject, str) else ""
        headers = header_fields(values)
        reply = any(
            values[name]["status"] == "present" for name in ("in_reply_to", "references")
        ) or any(headers[name]["status"] == "present" for name in ("in_reply_to", "references"))
        attachments = values["attachment_count"]
        features = frozenset(
            {
                ("folder", folder),
                ("year", date.year if isinstance(date, datetime) else "unknown"),
                ("direction", direction(values)),
                ("reply", reply or subject.startswith(("re:", "rép:"))),
                ("forward", subject.startswith(("fw:", "fwd:", "tr:"))),
                (
                    "attachments",
                    bool(attachments["value"]) if attachments["status"] == "present" else "unknown",
                ),
                (
                    "conversation",
                    any(
                        values[k]["status"] == "present"
                        for k in ("conversation_id", "conversation_index")
                    ),
                ),
                ("format", format_name),
            }
        )
        candidate = Candidate(folder, index, self.rng.random(), features)
        if len(self.uniform) < self.count:
            self.uniform.append(candidate)
        else:
            replacement = self.rng.randrange(self.seen)
            if replacement < self.count:
                self.uniform[replacement] = candidate
        for feature in features:
            previous = self.strata.get(feature)
            if previous is None or candidate.priority < previous.priority:
                self.strata[feature] = candidate

    def select(self) -> list[Candidate]:
        candidates = {(c.folder, c.index): c for c in [*self.uniform, *self.strata.values()]}
        covered: set[tuple[str, Any]] = set()
        chosen = []
        while candidates and len(chosen) < self.count:
            best = max(
                candidates.values(),
                key=lambda c: (len(c.features - covered), -c.priority, -c.folder, -c.index),
            )
            if not best.features - covered:
                break
            chosen.append(best)
            covered.update(best.features)
            del candidates[(best.folder, best.index)]
        for candidate in sorted(self.uniform, key=lambda c: c.priority):
            if len(chosen) >= self.count:
                break
            if candidate not in chosen:
                chosen.append(candidate)
        return sorted(chosen, key=lambda c: (c.folder, c.index))


def folders(root: Any, errors: Counter[str]) -> Iterator[dict[str, Any]]:
    # One live frame per depth; don't recursively build all native folder objects in RAM.
    stack = [(root, (), [], None)]
    seen: set[int] = set()
    while stack:
        item, locator, parent_names, parent = stack.pop()
        identifier = attempt(lambda: item.identifier)["value"]
        if identifier is not None:
            if identifier in seen:
                errors["folder_cycle"] += 1
                continue
            seen.add(identifier)
        name = attempt(lambda: item.name)
        names = [
            *parent_names,
            name["value"] if isinstance(name["value"], str) else "<unavailable>",
        ]
        counts = {
            "item_count": attempt(lambda: item.number_of_sub_messages),
            "subfolder_count": attempt(lambda: item.number_of_sub_folders),
        }
        for value in [name, *counts.values()]:
            if value["status"] in ("error", "unsupported"):
                errors["folder_metadata"] += 1
        yield {"locator": locator, "name": name, "path": names, "parent": parent, **counts}
        # A generator frame enumerates children lazily, retaining only ancestors.
        children = counts["subfolder_count"]["value"] or 0
        if children:
            yield from folder_children(item, locator, names, children, seen, errors)


def folder_children(
    item: Any,
    locator: tuple[int, ...],
    names: list[str],
    count: int,
    seen: set[int],
    errors: Counter[str],
) -> Iterator[dict[str, Any]]:
    # Explicit iterator stack keeps deep folder trees below Python's recursion limit.
    pending = [(item, locator, names, iter(range(count)))]
    while pending:
        parent, parent_locator, parent_names, indices = pending[-1]
        index = next(indices, None)
        if index is None:
            pending.pop()
            continue
        try:
            child = parent.get_sub_folder(index)
            identifier = child.identifier
            if identifier is not None and identifier in seen:
                errors["folder_cycle"] += 1
                continue
            if identifier is not None:
                seen.add(identifier)
            child_locator = (*parent_locator, index)
            name = attempt(lambda: child.name)
            path = [
                *parent_names,
                name["value"] if isinstance(name["value"], str) else "<unavailable>",
            ]
            items = attempt(lambda: child.number_of_sub_messages)
            subfolders = attempt(lambda: child.number_of_sub_folders)
            for value in (name, items, subfolders):
                if value["status"] in ("error", "unsupported"):
                    errors["folder_metadata"] += 1
            yield {
                "locator": child_locator,
                "name": name,
                "path": path,
                "parent": parent_locator,
                "item_count": items,
                "subfolder_count": subfolders,
            }
            pending.append((child, child_locator, path, iter(range(subfolders["value"] or 0))))
        except Exception:
            errors["folder_unreadable"] += 1


def resolve_folder(root: Any, locator: tuple[int, ...]) -> Any:
    for index in locator:
        root = root.get_sub_folder(index)
    return root


def shape(value: Any) -> dict[str, Any]:
    # No source-derived keys, strings, numeric values, hashes, or identifiers cross this boundary.
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        return {"type": "number"}
    if isinstance(value, datetime):
        return {"type": "datetime"}
    if isinstance(value, (str, bytes)):
        return {"type": "bytes" if isinstance(value, bytes) else "string", "length": len(value)}
    if isinstance(value, list):
        return {"type": "array", "length": len(value), "items": [shape(v) for v in value]}
    if isinstance(value, dict):
        return {"type": "object", "length": len(value)}
    return {"type": "unsupported"}


def field_shape(value: dict[str, Any]) -> dict[str, Any]:
    return {"status": value["status"], **shape(value["value"])}


def table_shape(value: dict[str, Any], allowed: dict[str, int]) -> dict[str, Any]:
    result = field_shape(value)
    if value["status"] in ("present", "empty"):
        result["items"] = [
            (
                {name: field_shape(row[name]) for name in allowed if name in row}
                if "item" not in row
                else {"item": {"status": "error"}}
            )
            for row in value["value"]
        ]
    return result


def structural_sample(sample: dict[str, Any], ordinal: int) -> dict[str, Any]:
    result = {
        "sample": ordinal,
        "fields": {name: field_shape(value) for name, value in sample["fields"].items()},
        "body": field_shape(sample["body"]["text"]),
        "body_properties": {
            name: {"status": value["status"]} for name, value in sample["body_properties"].items()
        },
        "body_reads": sample["body"]["availability"],
        "body_format": sample["body_format"],
        "body_source": sample["body"]["source"],
        "direction": sample["direction"],
        "attachments": table_shape(sample["attachments"], ATTACHMENT_PROPERTIES),
        "recipients": table_shape(sample["recipients"], RECIPIENT_PROPERTIES),
        "headers": {name: field_shape(value) for name, value in sample["headers"].items()},
    }
    if "character_length" in sample["body"]:
        result["body"]["original_character_length"] = sample["body"]["character_length"]
        result["body"]["truncated"] = sample["body"]["truncated"]
    return result


def validate_paths(pst: Path, output: Path) -> tuple[Path, Path]:
    source = pst.expanduser().resolve(strict=True)
    destination = output.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".pst":
        raise ValueError("Source must be a regular .pst file")
    if destination == REPO_ROOT or destination.is_relative_to(REPO_ROOT):
        raise ValueError("Reports must be outside the repository")
    if any((ancestor / ".git").exists() for ancestor in (destination, *destination.parents)):
        raise ValueError("Reports must be outside all Git repositories")
    if destination.exists() and not destination.is_dir():
        raise ValueError("Output must be a directory")
    for name in REPORTS:
        path = destination / name
        if path.is_symlink() or path.exists():
            if path.exists() and path.samefile(source):
                raise ValueError("Report path collides with source PST")
            raise ValueError("Output contains an existing report; choose a new directory")
        if path.resolve() == source:
            raise ValueError("Report path collides with source PST")
    return source, destination


def default_output() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return base / "JARVIS" / "PST_Analysis" / stamp


def count_states(target: dict[str, Counter[str]], values: dict[str, Any]) -> None:
    for name, value in values.items():
        target.setdefault(name, Counter({state: 0 for state in STATES}))[value["status"]] += 1


def markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# PST discovery inventory (private)",
        "",
        f"Status: **{inventory['status']}**. Mode: {inventory['mode']}.",
        f"Source bytes: {inventory['source']['size_bytes']}.",
        f"Visited items: {inventory['items_scanned']}; emails: {inventory['email_count']}.",
        f"Folder tree finished: {inventory['folder_tree_complete']}.",
        f"Metadata scope: {inventory['scope']}.",
        f"Earliest date: {inventory['earliest_date']}; latest date: {inventory['latest_date']}.",
        f"Direction evidence: {dict(inventory['directions'])}.",
        f"Body property availability (values not read): {dict(inventory['body_formats'])}.",
        f"Emails with attachments: {inventory['with_attachments']}.",
        f"Item classifications: {dict(inventory['item_kinds'])}.",
        f"Errors (counts only): {dict(inventory['errors'])}.",
        "",
        "Dates, counts and missing-field rates apply only to inspected emails.",
        "Sent/received classification is evidence, not verified mailbox ownership.",
        "Folder names and dates are private; only pst_samples_structure.json is shareable.",
        "",
        "## Folders",
        "",
        "| Folder path (JSON components) | Declared items | Inspected | Emails |",
        "| --- | ---: | ---: | ---: |",
    ]
    for folder in inventory["folders"]:
        path = json.dumps(folder["path"], ensure_ascii=True).replace("|", "&#124;")
        lines.append(
            f"| {path} | {folder['item_count']['value']} | "
            f"{folder['scanned']} | {folder['emails']} |"
        )
    lines.extend(
        [
            "",
            "## Metadata states",
            "",
            "| Field | Present | Empty | Absent | Unsupported | Error |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, counts in inventory["field_states"].items():
        lines.append(f"| {name} | " + " | ".join(str(counts[state]) for state in STATES) + " |")
    return "\n".join(lines) + "\n"


def inspect(
    pst: Path,
    output: Path,
    *,
    mode: str = "quick",
    max_items: int = 3000,
    max_samples: int = 150,
    seed: int = 42,
    max_body_chars: int = 200000,
    progress_every: int = 500,
    parser_module: Any = None,
    progress: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> dict[str, Any]:
    if mode not in ("quick", "full") or min(max_items, progress_every) < 1:
        raise ValueError("Invalid inspection limits")
    if min(max_samples, max_body_chars) < 0:
        raise ValueError("Sample and body limits must be nonnegative")
    source, destination = validate_paths(pst, output)
    if parser_module is None:
        try:
            import pypff as parser_module
        except ImportError:
            raise ValueError(
                "Install tools/requirements-pst.txt in an isolated discovery environment"
            ) from None
    inventory: dict[str, Any] = {
        "schema_version": 1,
        "mode": mode,
        "seed": seed,
        "status": "running",
        "scope": "inspected items only",
        "folder_tree_complete": False,
        "source": {"size_bytes": source.stat().st_size},
        "parser_version": attempt(lambda: parser_module.get_version()),
        "items_scanned": 0,
        "email_count": 0,
        "with_attachments": 0,
        "earliest_date": None,
        "latest_date": None,
        "date_basis": "all available client-submit and delivery timestamps (UTC)",
        "folders": [],
        "field_states": {},
        "property_tags": {},
        "directions": Counter(),
        "body_formats": Counter(),
        "item_kinds": Counter(),
        "item_classes": Counter(),
        "errors": Counter(),
        "sample_errors": Counter(),
    }
    errors = inventory["errors"]
    sampler = Sampler(max_samples, seed)
    samples: list[dict[str, Any]] = []
    archive = parser_module.file()
    before = source.stat()
    destination.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        reports = {
            name: stack.enter_context((destination / name).open("x", encoding="utf-8"))
            for name in REPORTS
        }
        # Supplying a Python rb handle prevents write access even inside the native parser.
        handle = stack.enter_context(source.open("rb"))
        opened = False
        try:
            progress("Opening archive read-only; enumerating folders.")
            header = handle.read(12)
            handle.seek(0)
            archive.open_file_object(handle, mode="r")
            opened = True
            for name in ("content_type", "encryption_type", "ascii_codepage"):
                inventory["source"][name] = attempt(lambda: getattr(archive, name))
            inventory["source"]["pff_signature"] = header[:4] == b"!BDN"
            inventory["source"]["format_version"] = (
                int.from_bytes(header[10:12], "little") if len(header) == 12 else None
            )
            root = archive.get_root_folder()
            if root is None:
                raise ValueError("Missing root folder")
            for folder in folders(root, errors):
                folder.update(scanned=0, emails=0)
                inventory["folders"].append(folder)
                if len(inventory["folders"]) % 100 == 0:
                    progress(f"Enumerated {len(inventory['folders'])} folders.")
            inventory["folder_tree_complete"] = not errors
            total = sum(f["item_count"]["value"] or 0 for f in inventory["folders"])
            inventory["declared_items"] = total
            selected = None
            if mode == "quick" and total > max_items:
                selected = iter(sorted(random.Random(seed).sample(range(total), max_items)))
                inventory["scope"] = "random limited subset; counts are not archive totals"
            else:
                inventory["scope"] = "all accessible folder items; no orphan/recovery scan"
            next_selected = next(selected, None) if selected is not None else None
            offset = 0
            progress(f"Folder enumeration finished; {total} declared items. Scanning metadata.")
            for folder_number, folder in enumerate(inventory["folders"]):
                count = folder["item_count"]["value"] or 0
                item_indices: Any = range(count)
                if selected is not None:
                    indices = []
                    while next_selected is not None and next_selected < offset + count:
                        indices.append(next_selected - offset)
                        next_selected = next(selected, None)
                    item_indices = indices
                offset += count
                try:
                    native_folder = resolve_folder(root, folder["locator"])
                except Exception:
                    errors["folder_reopen"] += 1
                    continue
                for index in item_indices:
                    inventory["items_scanned"] += 1
                    folder["scanned"] += 1
                    try:
                        message = native_folder.get_sub_message(index)
                        values, record = metadata(message)
                        kind = email_kind(values)
                        inventory["item_kinds"][kind] += 1
                        item_class = values["message_class"]["value"]
                        if isinstance(item_class, str):
                            inventory["item_classes"][item_class] += 1
                        if kind != "email":
                            if (
                                record.status != "present"
                                or values["message_class"]["status"] == "error"
                            ):
                                errors["item_metadata"] += 1
                            continue
                        inventory["email_count"] += 1
                        folder["emails"] += 1
                        body = {name: record.presence(tag) for name, tag in BODY_PROPERTIES.items()}
                        values.update({f"body_{name}": state for name, state in body.items()})
                        count_states(inventory["field_states"], values)
                        failures = sum(v["status"] == "error" for v in values.values())
                        if failures:
                            errors["metadata_fields"] += failures
                            errors["messages_with_metadata_errors"] += 1
                        for tag, value_type in record.types.items():
                            key = f"0x{tag:04X}/0x{value_type:04X}"
                            inventory["property_tags"][key] = (
                                inventory["property_tags"].get(key, 0) + 1
                            )
                        format_name = body_format(body)
                        inventory["body_formats"][format_name] += 1
                        inventory["directions"][direction(values)] += 1
                        if (values["attachment_count"]["value"] or 0) > 0:
                            inventory["with_attachments"] += 1
                        for name in ("sent_time", "received_time"):
                            date = values[name]["value"]
                            if isinstance(date, datetime):
                                stamp = serialize(date)
                                low, high = inventory["earliest_date"], inventory["latest_date"]
                                inventory["earliest_date"] = min(low, stamp) if low else stamp
                                inventory["latest_date"] = max(high, stamp) if high else stamp
                        sampler.add(folder_number, index, values, format_name)
                    except Exception:
                        errors["unreadable_items"] += 1
                    finally:
                        if inventory["items_scanned"] % progress_every == 0:
                            progress(
                                f"Inspected {inventory['items_scanned']} items; "
                                f"{inventory['email_count']} emails."
                            )
            inventory["status"] = (
                "partial_errors" if errors else "limited" if selected is not None else "complete"
            )
            progress(
                "Metadata scan finished; reading selected sample bodies and attachment metadata."
            )
            for candidate in sampler.select():
                try:
                    folder = inventory["folders"][candidate.folder]
                    native = resolve_folder(root, folder["locator"])
                    message = native.get_sub_message(candidate.index)
                    values, record = metadata(message)
                    body = {name: record.presence(tag) for name, tag in BODY_PROPERTIES.items()}
                    samples.append(
                        {
                            "folder_path": folder["path"],
                            "folder_locator": folder["locator"],
                            "item_index": candidate.index,
                            "fields": values,
                            "headers": header_fields(values),
                            "direction": direction(values),
                            "body_format": body_format(body),
                            "body_properties": body,
                            "body": sample_body(message, values, max_body_chars),
                            "recipients": recipient_metadata(message),
                            "attachments": attachment_metadata(message, values["attachment_count"]),
                        }
                    )
                    sample = samples[-1]
                    failures = sum(v["status"] == "error" for v in values.values())
                    failures += sum(
                        v["status"] == "error" for v in sample["body"]["availability"].values()
                    )
                    for name in ("recipients", "attachments"):
                        table = sample[name]
                        failures += table["status"] == "error"
                        for row in table["value"] or []:
                            failures += sum(v["status"] == "error" for v in row.values())
                    if failures:
                        inventory["sample_errors"]["field_failures"] += failures
                    if len(samples) % 25 == 0:
                        progress(f"Read {len(samples)} selected samples.")
                except Exception:
                    inventory["sample_errors"]["unreadable_samples"] += 1
        except KeyboardInterrupt:
            inventory["status"] = "interrupted"
            progress("Interrupted; writing partial reports from completed work.")
        except Exception:
            inventory["status"] = "failed"
            errors["archive_failure"] += 1
        finally:
            if opened:
                try:
                    archive.close()
                except Exception:
                    errors["archive_close"] += 1
            if errors and inventory["status"] in ("complete", "limited"):
                inventory["status"] = "partial_errors"
            after = source.stat()
            inventory["source"]["size_mtime_unchanged"] = (
                before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
            )
            if not inventory["source"]["size_mtime_unchanged"]:
                inventory["status"] = "source_changed_externally"
            inventory["samples_requested"] = max_samples
            inventory["samples_written"] = len(samples)
            inventory["sampling"] = (
                "greedy marginal-feature coverage plus uniform reservoir fill; "
                "not statistically weighted"
            )
            if inventory["sample_errors"] and inventory["status"] in ("complete", "limited"):
                inventory["status"] = "partial_sample_errors"
            sample_states: dict[str, Counter[str]] = {}
            for sample in samples:
                count_states(sample_states, sample["fields"])
            lengths = [
                s["body"]["character_length"] for s in samples if "character_length" in s["body"]
            ]
            # Dynamic folder names, classes and parser version are deliberately excluded.
            structure = {
                "schema_version": 1,
                "status": inventory["status"],
                "samples_written": len(samples),
                "inventory_field_states": inventory["field_states"],
                "sample_field_states": sample_states,
                "body_length_statistics": {
                    "count": len(lengths),
                    "min": min(lengths) if lengths else None,
                    "max": max(lengths) if lengths else None,
                    "mean": sum(lengths) / len(lengths) if lengths else None,
                },
                "property_tag_counts": inventory["property_tags"],
                "body_formats": inventory["body_formats"],
                "directions": inventory["directions"],
                "samples": [structural_sample(s, i + 1) for i, s in enumerate(samples)],
            }
            json.dump(serialize(inventory), reports[REPORTS[0]], indent=2, ensure_ascii=True)
            reports[REPORTS[1]].write(markdown(inventory))
            json.dump(
                serialize({"status": inventory["status"], "samples": samples}),
                reports[REPORTS[2]],
                indent=2,
                ensure_ascii=True,
            )
            json.dump(structure, reports[REPORTS[3]], indent=2, ensure_ascii=True)
    progress(f"Reports written. Status: {inventory['status']}; samples: {len(samples)}.")
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local read-only PST discovery; no JARVIS ingestion."
    )
    parser.add_argument("--pst", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument(
        "--max-items", type=int, default=3000, help="Quick-mode metadata item limit"
    )
    parser.add_argument("--max-samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-body-chars", type=int, default=200000)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args(argv)
    try:
        result = inspect(
            args.pst,
            args.output_dir or default_output(),
            mode=args.mode,
            max_items=args.max_items,
            max_samples=args.max_samples,
            seed=args.seed,
            max_body_chars=args.max_body_chars,
            progress_every=args.progress_every,
        )
    except Exception:
        print(
            "Inspection could not start or write reports. "
            "Check paths, permissions, limits and parser installation. "
            "Existing reports are never overwritten. See docs/pst-discovery.md.",
            file=sys.stderr,
        )
        return 2
    if result["status"] == "interrupted":
        return 130
    return 0 if result["status"] in ("complete", "limited") and not result["sample_errors"] else 1
