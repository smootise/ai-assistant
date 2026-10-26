"""Offline research only: signature spans, reviewed metrics, and private operator export."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
import logging
from pathlib import Path
import platform
import random
import re
import socket
import sqlite3
import sys
import time
import tracemalloc
from typing import Any, Callable
from unittest.mock import patch

from jarvis.email_preparation import (
    BUSINESS,
    PREPARATION_VERSION,
    PreparationOptions,
    prepare_email,
)

Span = tuple[int, int]
SCHEMA = "signature-benchmark-v1"
DETECTORS = (
    "baseline",
    "mail-reply",
    "melusine",
    "talon-brute",
    "talon-ml",
    "corpus",
    "hybrid",
    "baseline-hybrid",
)
SIGNOFF = re.compile(
    r"^(?:cordialement|bien cordialement|bien à vous|regards|best regards|--)[,! ]*$", re.I
)
ADDRESS = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
PHONE = re.compile(r"\+?\d[\d ().-]{7,}\d")


@dataclass
class Detection:
    spans: list[Span]
    unmapped: int = 0


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sender_identity(record: dict[str, Any]) -> str:
    address = ADDRESS.search(record.get("sender", ""))
    return address[0].casefold() if address else ""


def spans(value: Any, length: int) -> list[Span]:
    if not isinstance(value, list):
        raise ValueError("Invalid spans")
    result: list[Span] = []
    for pair in value:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(number) is not int for number in pair)
            or not 0 <= pair[0] < pair[1] <= length
        ):
            raise ValueError("Invalid span")
        result.append((pair[0], pair[1]))
    result.sort()
    if any(a[1] > b[0] for a, b in zip(result, result[1:])):
        raise ValueError("Overlapping spans")
    return result


def load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected record list")
    identifiers: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid record")
        if not isinstance(record.get("id"), str) or record["id"] in identifiers:
            raise ValueError("Invalid or duplicate identity")
        identifiers.add(record["id"])
        if not isinstance(record.get("body"), str):
            raise ValueError("Missing body")
        if not all(isinstance(record.get(key, ""), str) for key in ("sender", "format")):
            raise ValueError("Invalid metadata")
        label = record.get("label", {})
        if label.get("status") == "reviewed":
            if not label.get("reviewer") or label.get("body_sha256") != digest(record["body"]):
                raise ValueError("Reviewed label requires reviewer and matching body hash")
            gold = spans(label.get("signatures"), len(record["body"]))
            critical = spans(label.get("critical", []), len(record["body"]))
            if overlap(gold, critical):
                raise ValueError("Critical content overlaps signature label")
    return records


def canonical(text: str) -> tuple[str, list[int]]:
    # Only line-edge whitespace/CRLF normalization, never fuzzy or semantic alignment.
    chars: list[str] = []
    offsets: list[int] = []
    position = 0
    for raw in text.splitlines(keepends=True):
        value = raw.rstrip("\r\n")
        left = len(value) - len(value.lstrip())
        right = len(value.rstrip())
        for index in range(left, right):
            chars.append(value[index])
            offsets.append(position + index)
        if raw.endswith(("\r", "\n")):
            chars.append("\n")
            offsets.append(position + len(raw) - 1)
        position += len(raw)
    return "".join(chars), offsets


def align(text: str, candidates: list[str]) -> Detection:
    normalized, mapping = canonical(text)
    found: list[Span] = []
    unmapped = 0
    for candidate in candidates:
        value = canonical(candidate)[0].strip()
        if not value:
            continue
        start = normalized.find(value)
        if start < 0 or normalized.find(value, start + 1) >= 0:
            unmapped += 1
            continue
        found.append((mapping[start], mapping[start + len(value) - 1] + 1))
    return Detection(merge(found), unmapped)


def merge(ranges: list[Span]) -> list[Span]:
    result: list[Span] = []
    for start, end in sorted(set(ranges)):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap(left: list[Span], right: list[Span]) -> int:
    return sum(max(0, min(b, d) - max(a, c)) for a, b in left for c, d in right)


def baseline(record: dict[str, Any]) -> Detection:
    body = record["body"]
    sections = record.get("sections") or [
        {
            "start": 0,
            "end": len(body),
            "text": body,
            "kind": "new",
            "included": True,
            "confidence": "default",
        }
    ]
    prepared = prepare_email(
        {"normalized_text": body, "sections_json": json.dumps(sections)}, PreparationOptions()
    )
    return Detection(
        merge(
            [
                (section["start"], section["end"])
                for section in prepared.sections
                if section["excluded"] and section["kind"] == "signature"
            ]
        )
    )


def tail_candidate(record: dict[str, Any]) -> tuple[Span, str] | None:
    body = record["body"]
    # Deliberately abstain on quoted/forwarded histories: sender attribution is uncertain.
    if re.search(r"(?im)^\s*(?:>|De\s*:|From\s*:|On .+wrote:|Le .+écrit|-----)", body):
        return None
    matches = list(re.finditer(r"(?m)^.*$", body))
    for match in matches[-18:]:
        if not SIGNOFF.fullmatch(match[0].strip()):
            continue
        tail = body[match.start() :].rstrip()
        contacts = ADDRESS.findall(tail)
        sender = ADDRESS.search(record.get("sender", ""))
        if (
            not sender
            or sender[0].casefold() not in {v.casefold() for v in contacts}
            or not PHONE.search(tail)
            or BUSINESS.search(tail)
            or len(tail) > 1200
            or len(tail.splitlines()) > 16
        ):
            continue
        key = re.sub(r"(?<=\d)[ ().-]+(?=\d)", "", tail.casefold())
        key = re.sub(r"\s+", " ", key)
        return (match.start(), match.start() + len(tail)), key
    return None


def corpus_detector(reference: list[dict[str, Any]]) -> Callable[[dict[str, Any]], Detection]:
    support: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in reference:
        candidate = tail_candidate(record)
        if candidate:
            support[(sender_identity(record), candidate[1])].add(digest(record["body"]))

    def detect(record: dict[str, Any]) -> Detection:
        candidate = tail_candidate(record)
        if not candidate:
            return Detection([])
        evidence = support[(sender_identity(record), candidate[1])]
        # Three different reference bodies; evaluation body never counts toward support.
        return Detection([candidate[0]] if len(evidence - {digest(record["body"])}) >= 3 else [])

    return detect


def build_detector(
    name: str, reference: list[dict[str, Any]]
) -> Callable[[dict[str, Any]], Detection]:
    if name == "baseline":
        return baseline
    if name == "corpus":
        return corpus_detector(reference)
    if name == "baseline-hybrid":
        agreed = build_detector("hybrid", reference)

        def baseline_hybrid(record: dict[str, Any]) -> Detection:
            a, b = baseline(record), agreed(record)
            return Detection(merge(a.spans + b.spans), a.unmapped + b.unmapped)

        return baseline_hybrid
    if name == "hybrid":
        library = build_detector("mail-reply", reference)
        repeated = corpus_detector(reference)

        def hybrid(record: dict[str, Any]) -> Detection:
            a, b = library(record), repeated(record)
            return Detection(
                merge(
                    [
                        (max(x, u), min(y, v))
                        for x, y in a.spans
                        for u, v in b.spans
                        if max(x, u) < min(y, v)
                    ]
                ),
                a.unmapped + b.unmapped,
            )

        return hybrid
    if name == "mail-reply":
        from mailparser_reply import EmailMessage

        def mail_reply(record: dict[str, Any]) -> Detection:
            parsed = EmailMessage(
                text=record["body"], languages=["fr", "en"], keep_hyphen_lists=True
            )
            return align(record["body"], [reply.signatures for reply in parsed.read().replies])

        return mail_reply
    if name in {"talon-brute", "talon-ml"}:
        if name == "talon-brute":
            from talon.signature.bruteforce import extract_signature

            def extract(record: dict[str, Any]) -> tuple[str, str | None]:
                return extract_signature(record["body"])

        else:
            import talon
            from talon import signature

            talon.init()

            def extract(record: dict[str, Any]) -> tuple[str, str | None]:
                return signature.extract(record["body"], sender=record.get("sender", ""))

        def talon_detect(record: dict[str, Any]) -> Detection:
            return align(record["body"], [extract(record)[1] or ""])

        return talon_detect
    if name == "melusine":
        return melusine_detector()
    raise ValueError("Unknown detector")


def melusine_detector() -> Callable[[dict[str, Any]], Detection]:
    from melusine.processors import ContentTagger, Segmenter

    tagger = ContentTagger()
    segmenter = Segmenter()

    def detect(record: dict[str, Any]) -> Detection:
        messages = tagger.tag_email(segmenter.segment_text(record["body"])) or []
        return align(
            record["body"],
            [
                part["base_text"]
                for message in messages
                for part in (message.tags or [])
                if part["base_tag"] == "SIGNATURE"
            ],
        )

    return detect


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics(pairs: list[tuple[dict[str, Any], Detection]]) -> dict[str, Any]:
    total = sum(len(record["body"]) for record, _ in pairs)
    predicted = sum(b - a for _, result in pairs for a, b in result.spans)
    tp = fp = fn = exact = gold_blocks = predicted_blocks = critical_chars = 0
    false_messages = missed_blocks = reviewed = critical_messages = 0
    for record, result in pairs:
        label = record.get("label", {})
        if label.get("status") != "reviewed":
            continue
        reviewed += 1
        gold = spans(label["signatures"], len(record["body"]))
        hits = overlap(gold, result.spans)
        wrong = sum(b - a for a, b in result.spans) - hits
        tp += hits
        fp += wrong
        fn += sum(b - a for a, b in gold) - hits
        exact += len(set(gold) & set(result.spans))
        gold_blocks += len(gold)
        predicted_blocks += len(result.spans)
        false_messages += bool(wrong)
        missed_blocks += sum(overlap([span], result.spans) == 0 for span in gold)
        critical_hit = overlap(result.spans, spans(label.get("critical", []), len(record["body"])))
        critical_chars += critical_hit
        critical_messages += bool(critical_hit)
    return {
        "successful_messages": len(pairs),
        "reviewed_messages": reviewed,
        "estimated_excluded_chars": predicted,
        "estimated_text_reduction": ratio(predicted, total),
        "true_positive_chars": tp,
        "false_positive_chars": fp,
        "missed_signature_chars": fn,
        "character_precision": ratio(tp, tp + fp),
        "character_recall": ratio(tp, tp + fn),
        "signature_chars_correct_percent": None if not tp + fn else 100 * tp / (tp + fn),
        "exact_block_precision": ratio(exact, predicted_blocks),
        "exact_block_recall": ratio(exact, gold_blocks),
        "reviewed_predicted_blocks": predicted_blocks,
        "reviewed_gold_blocks": gold_blocks,
        "exact_true_positive_blocks": exact,
        "completely_missed_blocks": missed_blocks,
        "false_positive_messages": false_messages,
        "critical_false_positive_chars": critical_chars,
        "critical_false_positive_messages": critical_messages,
        "unmapped_candidates": sum(result.unmapped for _, result in pairs),
    }


def blocked_network(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Network disabled during benchmark")


def process_peak_rss() -> int | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        return (
            counters.PeakWorkingSetSize
            if query(handle(), ctypes.byref(counters), counters.cb)
            else None
        )
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def benchmark(
    records: list[dict[str, Any]], names: list[str], reference: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if {digest(r["body"]) for r in records} & {digest(r["body"]) for r in reference}:
        raise ValueError("Reference and evaluation bodies must be disjoint")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "python": platform.python_version(),
        "platform": platform.system(),
        "dataset_sha256": digest(json.dumps(records, sort_keys=True, ensure_ascii=True)),
        "reference_sha256": digest(json.dumps(reference, sort_keys=True, ensure_ascii=True)),
        "messages": len(records),
        "reference_messages": len(reference),
        "detectors": {},
        "packages": {},
        "memory_measurement": "tracemalloc Python allocations, not total RSS",
        "rss_measurement": "process lifetime peak; run one detector per process for comparisons",
        "baseline_rules": PREPARATION_VERSION,
        "expected_candidate_sources": {
            "melusine": "PyPI 3.4.0",
            "mail-reply": "PyPI 1.36",
            "talon": "44d1fa5e5b511734f9ed2eeb727dc57218446c22",
        },
        "ground_truth": {
            "synthetic_authored": sum(
                r.get("label", {}).get("source") == "synthetic-authored" for r in records
            ),
            "manual_reviewed": sum(
                r.get("label", {}).get("source") == "manual-reviewed"
                and r.get("label", {}).get("status") == "reviewed"
                for r in records
            ),
            "reviewed_unspecified": sum(
                r.get("label", {}).get("status") == "reviewed"
                and "source" not in r.get("label", {})
                for r in records
            ),
        },
    }
    for package in ("jarvis", "melusine", "mail-parser-reply", "talon"):
        try:
            report["packages"][package] = version(package)
        except PackageNotFoundError:
            report["packages"][package] = None
    review = [{**record, "predictions": {}} for record in records]
    for name in names:
        pairs: list[tuple[dict[str, Any], Detection]] = []
        failed = 0
        tracemalloc.start()
        started = time.perf_counter()
        # Third-party debug/repr output can contain email text. Suppress all such output.
        old_logging_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        with (
            patch.object(socket.socket, "connect", blocked_network),
            patch.object(socket.socket, "connect_ex", blocked_network),
            patch.object(socket, "create_connection", blocked_network),
            patch.object(socket, "getaddrinfo", blocked_network),
            patch.object(socket, "gethostbyname", blocked_network),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            try:
                detector = build_detector(name, reference)
            except Exception:
                detector = None
            for index, record in enumerate(records):
                try:
                    if detector is None:
                        raise RuntimeError("Unavailable detector")
                    result = detector(record)
                    result.spans = spans(result.spans, len(record["body"]))
                    pairs.append((record, result))
                    review[index]["predictions"][name] = {
                        "spans": result.spans,
                        "unmapped": result.unmapped,
                        "status": "ok",
                    }
                except Exception:
                    failed += 1
                    review[index]["predictions"][name] = {"status": "detector_failed"}
        logging.disable(old_logging_level)
        elapsed = time.perf_counter() - started
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        report["detectors"][name] = {
            "status": "unavailable" if detector is None else ("partial" if failed else "ok"),
            "failed_messages": failed,
            "elapsed_seconds": elapsed,
            "python_peak_bytes": peak,
            "process_peak_rss_bytes": process_peak_rss(),
            **metrics(pairs),
        }
    report["disagreement_messages"] = sum(
        len(
            {
                json.dumps([value.get("spans"), value["status"]], sort_keys=True)
                for value in record["predictions"].values()
            }
        )
        > 1
        for record in review
    )
    return report, review


def private_destination(path: Path) -> Path:
    resolved = path.resolve()
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValueError("Private artifacts must be outside Git repositories")
    if not resolved.parent.is_dir() or resolved.exists():
        raise ValueError("Output needs existing private parent and a fresh filename")
    return resolved


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=True, indent=2)
        output.write("\n")


def stratum(record: dict[str, Any]) -> tuple[str, str, str, str]:
    length = len(record["body"])
    size = "short" if length < 500 else ("medium" if length < 3000 else "long")
    body = record["body"]
    kind = "new"
    if re.search(r"(?im)^\s*(?:De|From)\s*:|forwarded|transféré|original message", body):
        kind = "outlook-forward"
    elif re.search(r"(?m)^\s*>|(?i:écrit|wrote)\s*:", body):
        kind = "reply"
    sender = sender_identity(record) or record.get("sender", "").casefold()
    return sender, size, kind, record.get("format", "unknown")


def sample_records(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item["id"]):
        groups[stratum(record)].append(record)
    rng = random.Random(seed)
    buckets = [groups[key] for key in sorted(groups)]
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    chosen: list[dict[str, Any]] = []
    while buckets and len(chosen) < count:
        for bucket in buckets:
            if bucket and len(chosen) < count:
                chosen.append(bucket.pop())
        buckets = [bucket for bucket in buckets if bucket]
    return chosen


def read_processed(db: Path, profile: str) -> list[dict[str, Any]]:
    if not db.is_file():
        raise ValueError("Use existing database")
    # No store/config factory: mode=ro cannot create/migrate the source or change journal mode.
    uri = db.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("BEGIN")
        markers = dict(connection.execute("SELECT key, value FROM _jarvis_meta"))
        if (
            markers.get("email_profile_name") != profile
            or markers.get("email_processing_schema_version") != "1"
        ):
            raise ValueError("Profile mismatch")
        query = """SELECT e.email_id, p.normalized_text, p.sections_json,
                          e.metadata_json, e.body_source
                   FROM email_items e JOIN email_processed p USING(email_id)
                   JOIN email_archives a USING(archive_id)
                   WHERE e.status='imported' AND e.item_kind='message'
                     AND a.integrity_status='verified' AND p.processed_at IS NOT NULL
                     AND p.processing_version='email-clean-v1' AND p.error_code IS NULL
                   ORDER BY e.email_id"""
        records = []
        for identity, body, sections, metadata_json, source in connection.execute(query):
            metadata = json.loads(metadata_json)
            fields = metadata.get("fields", {})
            sender = next(
                (
                    fields.get(key, {}).get("value")
                    for key in ("sender_smtp_address", "sender_address")
                    if isinstance(fields.get(key, {}).get("value"), str)
                    and fields.get(key, {}).get("value")
                ),
                "",
            )
            record = {
                "id": identity,
                "body": body,
                "sections": json.loads(sections),
                "sender": sender,
                "format": source or "unknown",
                "label": {
                    "status": "unreviewed",
                    "source": "manual-reviewed",
                    "body_sha256": digest(body),
                    "signatures": [],
                    "critical": [],
                    "reviewer": "",
                },
            }
            records.append(record)
        return records


def export_sample(db: Path, profile: str, count: int, seed: int) -> list[dict[str, Any]]:
    if not 50 <= count <= 100:
        raise ValueError("Use sample size 50..100")
    return sample_records(read_processed(db, profile), count, seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--reference", type=Path)
    run.add_argument("--detectors", nargs="+", choices=DETECTORS, default=["baseline"])
    run.add_argument("--report", required=True, type=Path)
    run.add_argument("--private-review", type=Path)
    export = commands.add_parser("export-readonly")
    export.add_argument("--db", required=True, type=Path)
    export.add_argument("--profile", required=True)
    export.add_argument("--count", type=int, default=75)
    export.add_argument("--seed", type=int, default=20261026)
    export.add_argument("--private-output", required=True, type=Path)
    export.add_argument("--private-reference", type=Path)
    export.add_argument("--allow-private-read", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export-readonly":
            destination = private_destination(args.private_output)
            reference_destination = (
                private_destination(args.private_reference) if args.private_reference else None
            )
            if destination == reference_destination or not 50 <= args.count <= 100:
                raise ValueError("Use distinct outputs and sample size 50..100")
            population = read_processed(args.db, args.profile)
            records = sample_records(population, args.count, args.seed)
            excluded = {digest(record["body"]) for record in records}
            reference = (
                [
                    {key: record[key] for key in ("id", "body", "sender", "format")}
                    for record in population
                    if digest(record["body"]) not in excluded
                ]
                if reference_destination
                else []
            )
            write_json(destination, records)
            if reference_destination:
                write_json(reference_destination, reference)
            print(json.dumps({"status": "exported", "messages": len(records), "seed": args.seed}))
            return 0
        destination = private_destination(args.private_review) if args.private_review else None
        if destination and destination == args.report.resolve():
            raise ValueError("Report and review destinations must differ")
        if args.report.exists():
            raise ValueError("Report already exists")
        records = load_records(args.input)
        reference = load_records(args.reference) if args.reference else []
        report, review = benchmark(records, args.detectors, reference)
        if destination:
            write_json(destination, review)
        write_json(args.report, report)
        print(json.dumps({"status": "complete", "messages": len(records)}))
        return 1 if any(r["status"] != "ok" for r in report["detectors"].values()) else 0
    except Exception:
        print(
            "Benchmark refused or failed; check inputs, dependencies and fresh output paths.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
