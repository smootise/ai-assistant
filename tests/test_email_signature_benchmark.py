from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from typing import Any
from unittest.mock import patch

import pytest

from tools import email_signature_benchmark as bench
from tools.signature_synthetic import FR, annotated, fixtures


def test_fixture_annotations_are_reproducible() -> None:
    evaluation, reference = fixtures()
    assert evaluation == bench.load_records(Path("tests/fixtures/signature_benchmark.json"))
    assert reference == bench.load_records(Path("tests/fixtures/signature_reference.json"))
    assert len(evaluation) >= 30
    assert all("<S>" not in r["body"] and "<C>" not in r["body"] for r in evaluation)


@pytest.mark.parametrize("bad", [[[True, 2]], [[-1, 2]], [[1, 99]], [[2, 1]], [[0, 3], [2, 4]]])
def test_reject_bad_spans(bad: Any) -> None:
    with pytest.raises(ValueError):
        bench.spans(bad, 10)


def test_reviewed_labels_require_matching_hash(tmp_path: Path) -> None:
    record = annotated("a", "Business.\n<S>Regards,\nBob</S>")
    record["body"] += "changed"
    source = tmp_path / "labels.json"
    source.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError):
        bench.load_records(source)


def test_alignment_preserves_crlf_unicode_and_original_text() -> None:
    text = "Résumé 👋\r\n\r\n   Regards,  \r\n Bob Smith  \r\n"
    before = text
    result = bench.align(text, ["Regards,\nBob Smith"])
    assert result.unmapped == 0
    a, b = result.spans[0]
    assert text[a:b] == "Regards,  \r\n Bob Smith"
    assert text == before


def test_ambiguous_or_rewritten_library_output_is_not_guessed() -> None:
    assert bench.align("Bob\nBob", ["Bob"]).unmapped == 1
    assert bench.align("Élodie", ["Elodie"]).unmapped == 1
    assert bench.align("Bob", [""]).spans == []


def test_metrics_distinguish_verified_labels_from_estimates() -> None:
    labeled = annotated("a", "<C>ABC</C><S>defg</S>")
    result = bench.metrics(
        [
            (labeled, bench.Detection([(2, 5)])),
            ({"body": "unknown", "id": "b"}, bench.Detection([(0, 7)])),
        ]
    )
    assert result["reviewed_messages"] == 1
    assert result["true_positive_chars"] == 2
    assert result["false_positive_chars"] == 1
    assert result["missed_signature_chars"] == 2
    assert result["critical_false_positive_chars"] == 1
    assert result["character_precision"] == pytest.approx(2 / 3)
    assert result["character_recall"] == 0.5
    assert result["estimated_excluded_chars"] == 10
    assert result["exact_block_precision"] == 0


def test_zero_denominators_are_undefined() -> None:
    result = bench.metrics([({"body": "", "id": "a"}, bench.Detection([]))])
    assert result["character_precision"] is None
    assert result["character_recall"] is None
    assert result["exact_block_recall"] is None


def test_corpus_requires_independent_support_and_sender_contact() -> None:
    evaluation, reference = fixtures()
    record = evaluation[0]
    assert bench.corpus_detector(reference[:2])(record).spans == []
    assert bench.corpus_detector(reference)(record).spans
    assert bench.corpus_detector([reference[0]] * 4)(record).spans == []
    wrong_sender = {**record, "sender": "other@example.invalid"}
    assert bench.corpus_detector(reference)(wrong_sender).spans == []
    quoted = {**record, "body": record["body"] + "\nFrom: Previous"}
    assert bench.corpus_detector(reference)(quoted).spans == []


def test_repetition_never_overrides_business_content() -> None:
    records = [
        {
            "id": str(i),
            "sender": "alice@example.invalid",
            "body": f"Hello {i}.\n\n{FR}\nOrder 25 items for 500 EUR.",
        }
        for i in range(5)
    ]
    assert bench.corpus_detector(records[:-1])(records[-1]).spans == []


def test_corpus_normalizes_telephone_format() -> None:
    evaluation, reference = fixtures()
    changed = deepcopy(evaluation[0])
    changed["body"] = changed["body"].replace("01 23 45 67 89", "01.23.45.67.89")
    assert bench.corpus_detector(reference)(changed).spans


def test_benchmark_rejects_reference_evaluation_leakage() -> None:
    evaluation, _ = fixtures()
    with pytest.raises(ValueError):
        bench.benchmark(evaluation, ["baseline"], evaluation[:1])


def test_baseline_matches_production_without_modifying_records() -> None:
    records, _ = fixtures()
    before = deepcopy(records)
    report, review = bench.benchmark(records, ["baseline"], [])
    assert records == before
    assert report["detectors"]["baseline"]["status"] == "ok"
    for record, inspected in zip(records, review):
        assert inspected["predictions"]["baseline"]["spans"] == bench.baseline(record).spans
        assert inspected["body"] == record["body"]
    assert "alice@example.invalid" not in json.dumps(report)
    assert "predictions" not in report


def test_failures_are_not_scored_as_success_and_diagnostics_are_private() -> None:
    records = [annotated("secret-id", "<C>secret-content</C>")]

    def fail(record: dict[str, Any]) -> bench.Detection:
        print(record["body"])
        raise ValueError(record["body"])

    with patch.object(bench, "build_detector", return_value=fail):
        report, review = bench.benchmark(records, ["baseline"], [])
    result = report["detectors"]["baseline"]
    assert result["status"] == "partial"
    assert result["successful_messages"] == 0
    assert result["reviewed_messages"] == 0
    assert result["failed_messages"] == 1
    assert "secret" not in json.dumps(report)
    assert review[0]["predictions"]["baseline"] == {"status": "detector_failed"}


def test_initialization_failure_is_reported() -> None:
    with patch.object(bench, "build_detector", side_effect=ImportError("private details")):
        report, _ = bench.benchmark([{"id": "a", "body": ""}], ["melusine"], [])
    assert report["detectors"]["melusine"]["status"] == "unavailable"
    assert "private details" not in json.dumps(report)


def test_network_is_blocked_during_detection() -> None:
    import socket

    def network(record: dict[str, Any]) -> bench.Detection:
        socket.create_connection(("example.invalid", 443))
        return bench.Detection([])

    with patch.object(bench, "build_detector", return_value=network):
        report, _ = bench.benchmark([{"id": "a", "body": ""}], ["baseline"], [])
    assert report["detectors"]["baseline"]["failed_messages"] == 1


def test_sampling_reproducible_across_order_and_diverse() -> None:
    records = [
        {
            "id": str(i),
            "body": "x" * (100 if i % 2 else 5000),
            "sender": f"sender{i % 10}",
            "format": "html" if i % 3 else "plain",
        }
        for i in range(150)
    ]
    selected = bench.sample_records(records, 75, 42)
    assert selected == bench.sample_records(list(reversed(records)), 75, 42)
    assert len({r["id"] for r in selected}) == 75
    assert len({r["sender"] for r in selected}) == 10
    assert {r["format"] for r in selected} == {"html", "plain"}
    assert {bench.stratum(r)[1] for r in selected} == {"short", "long"}


def synthetic_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE _jarvis_meta(key TEXT, value TEXT);
            INSERT INTO _jarvis_meta VALUES ('email_profile_name','test');
            INSERT INTO _jarvis_meta VALUES ('email_processing_schema_version','1');
            CREATE TABLE email_archives(archive_id TEXT, integrity_status TEXT);
            INSERT INTO email_archives VALUES ('a','verified');
            CREATE TABLE email_items(email_id TEXT, archive_id TEXT, status TEXT,
                item_kind TEXT, metadata_json TEXT, body_source TEXT);
            CREATE TABLE email_processed(email_id TEXT, normalized_text TEXT, sections_json TEXT,
                processed_at TEXT, processing_version TEXT, error_code TEXT);
        """)
        for index in range(120):
            connection.execute(
                "INSERT INTO email_items VALUES (?,?,?,?,?,?)",
                (
                    str(index),
                    "a",
                    "imported",
                    "message",
                    json.dumps(
                        {
                            "fields": {
                                "sender_smtp_address": {
                                    "status": "present",
                                    "value": f"sender{index % 5}@example.invalid",
                                }
                            }
                        }
                    ),
                    "plain",
                ),
            )
            connection.execute(
                "INSERT INTO email_processed VALUES (?,?,?,?,?,?)",
                (str(index), f"body {index}", "[]", "today", "email-clean-v1", None),
            )


def test_operator_export_is_readonly_and_profile_bound(tmp_path: Path) -> None:
    database = tmp_path / "synthetic.sqlite"
    synthetic_database(database)
    before = database.read_bytes()
    sample = bench.export_sample(database, "test", 75, 42)
    assert len(sample) == 75
    assert {r["sender"] for r in sample} == {f"sender{i}@example.invalid" for i in range(5)}
    assert database.read_bytes() == before
    with pytest.raises(ValueError):
        bench.export_sample(database, "wrong", 75, 42)
    assert database.read_bytes() == before
    absent = tmp_path / "absent.sqlite"
    with pytest.raises(ValueError):
        bench.export_sample(absent, "test", 75, 42)
    assert not absent.exists()


def test_operator_can_export_disjoint_unlabeled_reference(tmp_path: Path) -> None:
    database = tmp_path / "synthetic.sqlite"
    synthetic_database(database)
    before = database.read_bytes()
    evaluation, reference = tmp_path / "evaluation.json", tmp_path / "reference.json"
    assert (
        bench.main(
            [
                "export-readonly",
                "--db",
                str(database),
                "--profile",
                "test",
                "--count",
                "75",
                "--private-output",
                str(evaluation),
                "--private-reference",
                str(reference),
                "--allow-private-read",
            ]
        )
        == 0
    )
    sample, evidence = bench.load_records(evaluation), bench.load_records(reference)
    assert len(sample) == 75 and len(evidence) == 45
    assert not {r["body"] for r in sample} & {r["body"] for r in evidence}
    assert all("label" not in record for record in evidence)
    assert database.read_bytes() == before


def test_private_output_rejects_git_and_overwrite(tmp_path: Path) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()
    (repo / ".git").mkdir()
    with pytest.raises(ValueError):
        bench.private_destination(repo / "private.json")
    fresh = tmp_path / "fresh.json"
    bench.write_json(bench.private_destination(fresh), [])
    with pytest.raises(ValueError):
        bench.private_destination(fresh)


def test_cli_run_and_private_disagreements(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "aggregate.json"
    review = tmp_path / "review.json"
    assert (
        bench.main(
            [
                "run",
                "--input",
                "tests/fixtures/signature_benchmark.json",
                "--report",
                str(output),
                "--private-review",
                str(review),
            ]
        )
        == 0
    )
    assert "body" not in capsys.readouterr().out
    assert json.loads(review.read_text())[0]["body"]
    before = output.read_bytes()
    assert (
        bench.main(
            ["run", "--input", "tests/fixtures/signature_benchmark.json", "--report", str(output)]
        )
        == 1
    )
    assert before == output.read_bytes()


def test_hybrid_requires_agreement() -> None:
    evaluation, reference = fixtures()

    def library(record: dict[str, Any]) -> bench.Detection:
        return bench.Detection([(0, len(record["body"]))])

    original = bench.build_detector
    with patch.object(
        bench,
        "build_detector",
        side_effect=lambda name, refs: (library if name == "mail-reply" else original(name, refs)),
    ):
        hybrid = original("hybrid", reference)
    assert hybrid(evaluation[0]).spans == bench.corpus_detector(reference)(evaluation[0]).spans
    assert hybrid(evaluation[18]).spans == []
