"""Synthetic libpff contracts; no real mail, databases, or network access."""

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tools import pst_discovery as discovery
from tools.pst_parser import BODY_PROPERTIES, PROPERTIES, Record, field, metadata


class Entry:
    def __init__(self, tag: int, value: Any, value_type: int | None = None) -> None:
        self.entry_type = tag
        if value_type is None:
            value_type = (
                0x0040
                if isinstance(value, datetime)
                else (
                    0x0003
                    if isinstance(value, int)
                    else 0x0102 if isinstance(value, bytes) else 0x001F
                )
            )
        self.value_type = value_type
        self.value = value
        self.reads = 0

    def read(self) -> Any:
        self.reads += 1
        if self.entry_type in (*BODY_PROPERTIES.values(), 0x3701):
            raise AssertionError("Body or attachment data read through inventory adapter")
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value

    data = property(read)
    data_as_string = property(read)
    data_as_integer = property(read)
    data_as_boolean = property(read)
    data_as_datetime = property(read)


class Row:
    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries
        self.number_of_entries = len(entries)

    def get_entry(self, index: int) -> Entry:
        return self.entries[index]


class Item:
    def __init__(self, values: dict[int, Any]) -> None:
        self.row = Row([Entry(tag, value) for tag, value in values.items()])
        self.number_of_record_sets = 1

    def get_record_set(self, index: int) -> Row:
        assert index == 0
        return self.row


class Message(Item):
    def __init__(
        self,
        index: int = 0,
        *,
        values: dict[str, Any] | None = None,
        plain: Any = "PRIVATE_BODY_SENTINEL",
        html: Any = None,
        rtf: bool = False,
        attachment: bool = False,
    ) -> None:
        defaults = {
            "message_class": "IPM.Note",
            "subject": f"PRIVATE_SUBJECT_SENTINEL {index}",
            "sender_name": "PRIVATE_NAME_SENTINEL",
            "sender_smtp_address": "PRIVATE_ADDRESS_SENTINEL@example.invalid",
            "sent_time": datetime(2000 + index % 20, 1, 1, tzinfo=timezone.utc),
            "received_time": datetime(2000 + index % 20, 1, 2, tzinfo=timezone.utc),
            "message_flags": 0x20 if index % 2 else 0,
            "internet_message_id": f"<PRIVATE_ID_SENTINEL-{index}>",
            "conversation_id": b"PRIVATE_BINARY_SENTINEL" if index % 2 else None,
        }
        defaults.update(values or {})
        props = {PROPERTIES[name]: value for name, value in defaults.items() if value is not None}
        for name, value in (("plain", plain), ("html", html), ("rtf", b"RTF" if rtf else None)):
            if value is not None:
                props[BODY_PROPERTIES[name]] = value
        super().__init__(props)
        self.identifier = index + 10000
        self.number_of_attachments = int(attachment)
        self.body_reads = 0
        self.raw_plain = plain
        self.raw_html = html
        self.attachment = Item(
            {
                0x3707: "PRIVATE_FILENAME_SENTINEL.pdf",
                0x370E: "application/pdf",
                0x0E20: 999,
                0x3701: b"DO_NOT_READ",
            }
        )
        self.recipients = SimpleNamespace(
            number_of_recipients=1,
            get_recipient=lambda index: Row(
                [
                    Entry(0x3001, "PRIVATE_RECIPIENT_SENTINEL"),
                    Entry(0x3003, "recipient@example.invalid"),
                ]
            ),
        )

    @property
    def plain_text_body(self) -> Any:
        self.body_reads += 1
        if isinstance(self.raw_plain, Exception):
            raise self.raw_plain
        return self.raw_plain

    @property
    def html_body(self) -> Any:
        self.body_reads += 1
        if isinstance(self.raw_html, Exception):
            raise self.raw_html
        return self.raw_html

    @property
    def rtf_body(self) -> Any:
        raise AssertionError("RTF must never be loaded")

    def get_attachment(self, index: int) -> Item:
        assert index == 0
        return self.attachment


class Folder:
    def __init__(
        self, name: str, messages: list[Any] | None = None, children: list[Any] | None = None
    ) -> None:
        self.name = name
        self.messages = messages or []
        self.children = children or []
        self.number_of_sub_messages = len(self.messages)
        self.number_of_sub_folders = len(self.children)
        self.identifier = id(self)

    def get_sub_folder(self, index: int) -> Any:
        value = self.children[index]
        if isinstance(value, BaseException):
            raise value
        return value

    def get_sub_message(self, index: int) -> Any:
        value = self.messages[index]
        if isinstance(value, BaseException):
            raise value
        return value


class Archive:
    content_type = 112
    encryption_type = 1
    ascii_codepage = "cp1252"

    def __init__(self, root: Folder) -> None:
        self.root = root
        self.closed = False

    def open_file_object(self, handle: Any, mode: str) -> None:
        assert mode == "r"
        assert handle.mode == "rb"
        assert not handle.writable()
        self.handle = handle

    def get_root_folder(self) -> Folder:
        return self.root

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def run_inspection(tmp_path: Path) -> Any:
    source = tmp_path / "synthetic.pst"
    source.write_bytes(b"!BDN" + b"\0" * 6 + b"\x17\0" + b"synthetic")
    counter = 0

    def run(root: Folder, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], Archive]:
        nonlocal counter
        counter += 1
        output = tmp_path / f"reports-{counter}"
        archive = Archive(root)
        module = SimpleNamespace(file=lambda: archive, get_version=lambda: "synthetic")
        before = source.read_bytes(), source.stat().st_mtime_ns
        inventory = discovery.inspect(
            source, output, parser_module=module, progress=lambda _: None, **kwargs
        )
        assert before == (source.read_bytes(), source.stat().st_mtime_ns)
        assert archive.closed
        reports = {
            name: json.loads((output / name).read_text(encoding="utf-8"))
            for name in discovery.REPORTS
            if name.endswith(".json")
        }
        assert "PST discovery inventory" in (output / "pst_inventory.md").read_text()
        return inventory, reports, archive

    return run


def test_folder_tree_and_counts(run_inspection: Any) -> None:
    root = Folder("root", [Message()], [Folder("a", [Message(1)], [Folder("b", [Message(2)])])])
    result, _, _ = run_inspection(root, mode="full")
    assert [folder["path"] for folder in result["folders"]] == [
        ["root"],
        ["root", "a"],
        ["root", "a", "b"],
    ]
    assert [folder["parent"] for folder in result["folders"]] == [None, (), (0,)]
    assert result["email_count"] == 3
    assert result["status"] == "complete"
    assert result["source"]["format_version"] == 23
    assert result["earliest_date"].startswith("2000-")
    assert result["latest_date"].startswith("2002-")


def test_deep_folders_and_cycle() -> None:
    root = Folder("leaf")
    for _ in range(1100):
        root = Folder("branch", children=[root])
    errors = discovery.Counter()
    assert sum(1 for _ in discovery.folders(root, errors)) == 1101
    root.children[0] = root
    assert len(list(discovery.folders(root, errors))) == 1
    assert errors["folder_cycle"] == 1


def test_field_states_and_metadata() -> None:
    message = Message(
        values={"subject": "", "sender_name": OSError("SECRET"), "conversation_id": None}
    )
    values, _ = metadata(message)
    assert values["subject"] == field("")
    assert values["conversation_id"] == field()
    assert values["sender_name"] == field(status="error")
    assert values["internet_message_id"]["value"].startswith("<PRIVATE_ID")
    assert "SECRET" not in json.dumps(discovery.serialize(values))
    assert Record(object()).get(123)["status"] == "unsupported"


def test_record_enumeration_failure_is_not_absent() -> None:
    item = Mock(number_of_record_sets=1)
    item.get_record_set.side_effect = OSError("PRIVATE")
    assert Record(item).get(123)["status"] == "error"


def test_unhandled_property_type() -> None:
    row = Row([Entry(0x9999, "unknown", value_type=0x101F)])
    assert Record(row, record_set=True).get(0x9999)["status"] == "unsupported"


@pytest.mark.parametrize("plain", [None, b"", "", OSError("PRIVATE_BODY_ERROR")])
def test_html_fallback(plain: Any) -> None:
    message = Message(
        plain=plain,
        html=(
            b"<html><head>hidden</head><p>H&amp;I</p>"
            b"<script>secret</script><div>next</div></html>"
        ),
    )
    values, _ = metadata(message)
    body = discovery.sample_body(message, values, 1000)
    assert body["text"]["value"] == "H&I\nnext"
    assert body["source"] == "html"
    assert "html" in body["availability"]
    assert body["availability"]["plain"]["status"] == (
        "error" if isinstance(plain, Exception) else "absent" if plain is None else "empty"
    )


def test_body_truncation_encoding_and_rtf_only(run_inspection: Any) -> None:
    message = Message(plain=b"caf\xe9", values={"internet_codepage": 1252})
    body = discovery.sample_body(message, metadata(message)[0], 3)
    assert body["text"]["value"] == "caf"
    assert body["character_length"] == 4
    assert body["truncated"]
    assert body["decoding"] == "cp1252"
    result, reports, _ = run_inspection(Folder("root", [Message(plain=None, rtf=True)]))
    assert result["body_formats"] == {"rtf_only": 1}
    sample = reports["pst_samples_private.json"]["samples"][0]
    assert sample["body"]["text"]["status"] == "absent"


def test_inventory_never_reads_bodies_or_attachment_data(run_inspection: Any) -> None:
    message = Message(attachment=True, html="<p>HTML</p>")
    result, reports, _ = run_inspection(Folder("root", [message]), max_samples=0)
    assert message.body_reads == 0
    assert result["with_attachments"] == 1
    assert all(
        entry.reads == 0
        for entry in message.row.entries
        if entry.entry_type in BODY_PROPERTIES.values()
    )
    text = json.dumps(reports["pst_inventory.json"])
    assert "PRIVATE_BODY_SENTINEL" not in text
    assert "PRIVATE_SUBJECT_SENTINEL" not in text
    assert "HTML" not in text


def test_attachment_metadata_only(run_inspection: Any) -> None:
    message = Message(attachment=True)
    _, reports, _ = run_inspection(Folder("root", [message]))
    attachment = reports["pst_samples_private.json"]["samples"][0]["attachments"]["value"][0]
    assert attachment["filename"]["value"] == "PRIVATE_FILENAME_SENTINEL.pdf"
    assert attachment["size"]["value"] == 999
    assert (
        next(entry for entry in message.attachment.row.entries if entry.entry_type == 0x3701).reads
        == 0
    )


def test_sampling_is_deterministic_and_diverse(run_inspection: Any) -> None:
    messages = [Message(i) for i in range(100)]
    messages[-1] = Message(
        99, attachment=True, plain=None, html="<p>unique</p>", values={"subject": "Fwd: unique"}
    )
    root = Folder(
        "root", messages, [Folder("rare", [Message(8, values={"in_reply_to": "<parent>"})])]
    )
    _, reports, _ = run_inspection(root, max_samples=25, seed=8)
    _, repeated, _ = run_inspection(root, max_samples=25, seed=8)
    private = reports["pst_samples_private.json"]
    assert private == repeated["pst_samples_private.json"]
    samples = private["samples"]
    assert len(samples) == 25
    assert any(s["folder_path"] == ["root", "rare"] for s in samples)
    assert any(s["body_format"] == "html_only" for s in samples)
    assert {s["direction"] for s in samples} == {"sent", "received_evidence"}
    assert len({s["fields"]["sent_time"]["value"][:4] for s in samples}) >= 15
    assert any(s["fields"]["conversation_id"]["status"] == "absent" for s in samples)
    _, changed, _ = run_inspection(root, max_samples=25, seed=9)
    assert private != changed["pst_samples_private.json"]


def test_quick_mode_spans_folders_and_archive_positions(run_inspection: Any) -> None:
    root = Folder(
        "root",
        [Message(i) for i in range(100)],
        [Folder("tail", [Message(i) for i in range(100, 200)])],
    )
    result, reports, _ = run_inspection(root, max_items=30, max_samples=10)
    assert result["items_scanned"] == 30
    assert result["status"] == "limited"
    assert all(0 < f["scanned"] < 100 for f in result["folders"])
    assert any(s["item_index"] > 50 for s in reports["pst_samples_private.json"]["samples"])


def test_no_private_values_in_shareable_report_or_console(run_inspection: Any, capsys: Any) -> None:
    message = Message(1, attachment=True, html="PRIVATE_HTML_SENTINEL")
    root = Folder("PRIVATE_FOLDER_SENTINEL", [message])
    _, reports, _ = run_inspection(root)
    private = json.dumps(reports["pst_samples_private.json"])
    structure = json.dumps(reports["pst_samples_structure.json"])
    assert "PRIVATE_BODY_SENTINEL" in private
    for value in (
        "PRIVATE",
        "example.invalid",
        "2001-",
        b"PRIVATE_BINARY_SENTINEL".hex(),
        "application/pdf",
        "synthetic",
        "recipient@example.invalid",
    ):
        assert value not in structure
    sample = reports["pst_samples_structure.json"]["samples"][0]
    assert sample["fields"]["subject"]["type"] == "string"
    assert sample["attachments"]["items"][0]["filename"]["status"] == "present"
    assert capsys.readouterr().out == ""


def test_malformed_and_non_email_items(run_inspection: Any) -> None:
    root = Folder(
        "root",
        [
            OSError("PRIVATE"),
            Message(values={"message_class": "IPM.Appointment"}),
            Message(values={"message_class": "CUSTOM_PRIVATE"}),
            Message(values={"message_class": None}),
            Message(),
        ],
        [OSError("PRIVATE_FOLDER")],
    )
    result, reports, _ = run_inspection(root)
    assert result["status"] == "partial_errors"
    assert result["item_kinds"] == {"non_email": 1, "unsupported": 1, "unknown": 1, "email": 1}
    assert result["errors"]["unreadable_items"] == 1
    assert result["errors"]["folder_unreadable"] == 1
    assert not result["folder_tree_complete"]
    assert len(reports["pst_samples_private.json"]["samples"]) == 1
    assert "CUSTOM_PRIVATE" not in json.dumps(reports["pst_samples_structure.json"])


def test_interruption_writes_partial_reports(run_inspection: Any) -> None:
    result, reports, _ = run_inspection(
        Folder("root", [Message(), KeyboardInterrupt(), Message(1)])
    )
    assert result["status"] == "interrupted"
    assert result["email_count"] == 1
    assert reports["pst_samples_private.json"] == {"status": "interrupted", "samples": []}
    assert reports["pst_samples_structure.json"]["status"] == "interrupted"


def test_body_errors_mark_partial_samples(run_inspection: Any) -> None:
    result, reports, _ = run_inspection(
        Folder("root", [Message(plain=OSError("PRIVATE"), html=OSError("SECRET"))])
    )
    assert result["status"] == "partial_sample_errors"
    assert result["sample_errors"]["field_failures"] == 2
    assert "SECRET" not in json.dumps(reports)
    structural = reports["pst_samples_structure.json"]["samples"][0]
    assert structural["body_reads"]["plain"]["status"] == "error"


def test_sample_phase_interruption_preserves_completed_sample(run_inspection: Any) -> None:
    class InterruptBody(Message):
        @property
        def plain_text_body(self) -> str:
            raise KeyboardInterrupt()

    result, reports, _ = run_inspection(Folder("root", [Message(), InterruptBody(1)]))
    assert result["status"] == "interrupted"
    assert len(reports["pst_samples_private.json"]["samples"]) == 1


def test_close_failure_is_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.pst"
    source.write_bytes(b"synthetic")
    archive = Archive(Folder("root"))
    archive.close = Mock(side_effect=OSError("PRIVATE_CLOSE"))
    result = discovery.inspect(
        source,
        tmp_path / "reports",
        parser_module=SimpleNamespace(file=lambda: archive, get_version=lambda: "synthetic"),
        progress=lambda _: None,
    )
    assert result["status"] == "partial_errors"
    assert result["errors"]["archive_close"] == 1


def test_header_metadata_and_direction() -> None:
    values, _ = metadata(
        Message(
            values={
                "transport_headers": (
                    "Message-ID: <id>\nIn-Reply-To: <parent>\n" "References: <a> <b>\n\n"
                )
            }
        )
    )
    assert discovery.header_fields(values)["in_reply_to"]["value"] == ["<parent>"]
    assert discovery.direction(values) == "received_evidence"
    values["message_flags"] = field(8)
    assert discovery.direction(values) == "draft"
    values["message_flags"] = field(32)
    assert discovery.direction(values) == "sent"
    values["message_flags"] = field()
    values["received_time"] = field()
    assert discovery.direction(values) == "unknown"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_items": 0},
        {"max_samples": -1},
        {"max_body_chars": -1},
        {"progress_every": 0},
        {"mode": "bad"},
    ],
)
def test_invalid_limits(run_inspection: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        run_inspection(Folder("root"), **kwargs)


def test_safe_output_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.pst"
    source.write_bytes(b"source")
    with pytest.raises(ValueError, match="outside the repository"):
        discovery.validate_paths(source, discovery.REPO_ROOT / "reports")
    other_repo = tmp_path / "other"
    other_repo.mkdir()
    (other_repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ValueError, match="all Git"):
        discovery.validate_paths(source, other_repo / "reports")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "pst_samples_private.json").write_text("existing")
    with pytest.raises(ValueError, match="existing report"):
        discovery.validate_paths(source, reports)
    assert source.read_bytes() == b"source"
    assert (reports / "pst_samples_private.json").read_text() == "existing"


def test_hardlink_source_collision(tmp_path: Path) -> None:
    source = tmp_path / "source.pst"
    source.write_bytes(b"source")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "pst_inventory.json").hardlink_to(source)
    with pytest.raises(ValueError, match="collides"):
        discovery.validate_paths(source, reports)
    assert source.read_bytes() == b"source"


def test_archive_open_failure_writes_safe_failure_reports(tmp_path: Path) -> None:
    source = tmp_path / "source.pst"
    source.write_bytes(b"synthetic")
    archive = Mock()
    archive.open_file_object.side_effect = OSError("PRIVATE_FAILURE")
    module = SimpleNamespace(file=lambda: archive, get_version=lambda: "synthetic")
    output = tmp_path / "reports"
    result = discovery.inspect(source, output, parser_module=module, progress=lambda _: None)
    assert result["status"] == "failed"
    assert result["errors"] == {"archive_failure": 1}
    assert "PRIVATE_FAILURE" not in "".join(p.read_text() for p in output.iterdir())


def test_default_output_outside_repo_and_console_failure_is_private(
    monkeypatch: Any, capsys: Any
) -> None:
    assert not discovery.default_output().is_relative_to(discovery.REPO_ROOT)
    monkeypatch.setattr(discovery, "inspect", Mock(side_effect=OSError("PRIVATE_PATH")))
    assert discovery.main(["--pst", "PRIVATE_PATH.pst"]) == 2
    captured = capsys.readouterr()
    assert "PRIVATE_PATH" not in captured.err
    assert captured.out == ""
