"""Read-only libpff adapter. No JARVIS imports or external service calls."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

# Property IDs, not combined property/type tags. Body/attachment data is never decoded here.
PROPERTIES = {
    "message_class": 0x001A,
    "subject": 0x0037,
    "sender_name": 0x0C1A,
    "sender_address": 0x0C1F,
    "sender_address_type": 0x0C1E,
    "sender_smtp_address": 0x5D01,
    "display_to": 0x0E04,
    "display_cc": 0x0E03,
    "display_bcc": 0x0E02,
    "sent_time": 0x0039,
    "received_time": 0x0E06,
    "creation_time": 0x3007,
    "modification_time": 0x3008,
    "internet_message_id": 0x1035,
    "conversation_id": 0x3013,
    "conversation_topic": 0x0070,
    "conversation_index": 0x0071,
    "in_reply_to": 0x1042,
    "references": 0x1039,
    "transport_headers": 0x007D,
    "message_flags": 0x0E07,
    "message_status": 0x0E17,
    "message_size": 0x0E08,
    "importance": 0x0017,
    "sensitivity": 0x0036,
    "internet_codepage": 0x3FDE,
    "message_codepage": 0x3FFD,
}
BODY_PROPERTIES = {"plain": 0x1000, "html": 0x1013, "rtf": 0x1009}
ATTACHMENT_PROPERTIES = {
    "filename": 0x3707,
    "short_filename": 0x3704,
    "size": 0x0E20,
    "mime_type": 0x370E,
    "method": 0x3705,
    "content_id": 0x3712,
}
RECIPIENT_PROPERTIES = {
    "name": 0x3001,
    "address_type": 0x3002,
    "address": 0x3003,
    "smtp_address": 0x39FE,
    "recipient_type": 0x0C15,
}


def field(value: Any = None, status: str | None = None) -> dict[str, Any]:
    if status is None:
        status = "absent" if value is None else "empty" if value in ("", b"", []) else "present"
    return {"status": status, "value": value}


def attempt(operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        return field(operation())
    except AttributeError:
        return field(status="unsupported")
    except Exception:
        # Native exception messages can contain private property values or source paths.
        return field(status="error")


def serialize(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"encoding": "hex", "data": value.hex(), "byte_length": len(value)}
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value


class Record:
    def __init__(self, item: Any, record_set: bool = False) -> None:
        self.entries: dict[int, Any] = {}
        self.types: dict[int, int] = {}
        self.status = "present"
        try:
            sets = 1 if record_set else item.number_of_record_sets
            for index in range(sets):
                row = item if record_set else item.get_record_set(index)
                for entry_index in range(row.number_of_entries):
                    entry = row.get_entry(entry_index)
                    tag = entry.entry_type
                    self.entries[tag] = entry
                    self.types[tag] = entry.value_type
        except AttributeError:
            self.status = "unsupported"
        except Exception:
            self.status = "error"

    def presence(self, tag: int) -> dict[str, Any]:
        if tag in self.entries:
            return field(status="present")
        return field(status="absent" if self.status == "present" else self.status)

    def get(self, tag: int) -> dict[str, Any]:
        if tag not in self.entries:
            return self.presence(tag)
        entry = self.entries[tag]
        value_type = self.types[tag]
        getters = {
            0x0002: "data_as_integer",
            0x0003: "data_as_integer",
            0x0014: "data_as_integer",
            0x000B: "data_as_boolean",
            0x001E: "data_as_string",
            0x001F: "data_as_string",
            0x0040: "data_as_datetime",
            0x0102: "data",
            0x0048: "data",
        }
        getter = getters.get(value_type)
        if getter is None:
            return field(status="unsupported")
        return attempt(lambda: getattr(entry, getter))


def metadata(message: Any) -> tuple[dict[str, Any], Record]:
    record = Record(message)
    values = {name: record.get(tag) for name, tag in PROPERTIES.items()}
    values["attachment_count"] = attempt(lambda: message.number_of_attachments)
    values["item_identifier"] = attempt(lambda: message.identifier)
    return values, record


def recipient_metadata(message: Any) -> dict[str, Any]:
    recipients = attempt(lambda: message.recipients)
    if recipients["status"] != "present":
        return recipients
    table = recipients["value"]
    try:
        rows = []
        for index in range(table.number_of_recipients):
            row = Record(table.get_recipient(index), record_set=True)
            rows.append({name: row.get(tag) for name, tag in RECIPIENT_PROPERTIES.items()})
        return field(rows)
    except Exception:
        return field(status="error")


def attachment_metadata(message: Any, count: dict[str, Any]) -> dict[str, Any]:
    if count["status"] != "present":
        return field(status=count["status"])
    rows = []
    for index in range(count["value"]):
        try:
            attachment = message.get_attachment(index)
            record = Record(attachment)
            rows.append({name: record.get(tag) for name, tag in ATTACHMENT_PROPERTIES.items()})
        except Exception:
            rows.append({"item": field(status="error")})
    return field(rows)
