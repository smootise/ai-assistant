"""Compatibility exports for the standalone discovery utility."""

__all__ = [
    "PROPERTIES",
    "BODY_PROPERTIES",
    "ATTACHMENT_PROPERTIES",
    "RECIPIENT_PROPERTIES",
    "Record",
    "field",
    "attempt",
    "serialize",
    "metadata",
    "recipient_metadata",
    "attachment_metadata",
    "decode_text",
    "html_to_text",
    "header_fields",
    "email_kind",
]

from jarvis.pst import (
    PROPERTIES as PROPERTIES,
    BODY_PROPERTIES as BODY_PROPERTIES,
    ATTACHMENT_PROPERTIES as ATTACHMENT_PROPERTIES,
    RECIPIENT_PROPERTIES as RECIPIENT_PROPERTIES,
    Record as Record,
    field as field,
    attempt as attempt,
    serialize as serialize,
    metadata as metadata,
    recipient_metadata as recipient_metadata,
    attachment_metadata as attachment_metadata,
    decode_text as decode_text,
    html_to_text as html_to_text,
    header_fields as header_fields,
    email_kind as email_kind,
)
