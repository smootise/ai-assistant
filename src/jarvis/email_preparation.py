"""Deterministic, recoverable email preparation without model or vector clients."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
from itertools import groupby, islice
import json
import re
from typing import Any, Callable, TYPE_CHECKING

from jarvis.email_cleaning import BOILERPLATE, FORWARD, HEADER, MOBILE, REPLY, SIGNOFF, VALUABLE
from jarvis.email_cleaning import PROCESSING_VERSION
from jarvis.email_cleaning import _clean_urls

if TYPE_CHECKING:
    from jarvis.email_preparation_store import EmailPreparationStore

PREPARATION_VERSION = "email-prepare-v1"
TOKEN = re.compile(r"\w+(?:[-'’./@]\w+)*|[^\w\s]", re.UNICODE)
BUSINESS = re.compile(
    r"\b(?:command\w*|order\w*|prix|price\w*|devis|factur\w*|invoice\w*|"
    r"livrai\w*|deliver\w*|merci|please|veuillez|confirm\w*|valid\w*|approuv\w*|"
    r"payment\w*|paiement\w*|réf\w*|ref\w*|stock\w*|quantit\w*|deadline\w*|"
    r"instruction\w*|budget\w*|projet\w*|project\w*|contrat\w*|contract\w*)\b",
    re.I,
)
NAME = re.compile(
    r"[A-ZÀ-ÖØ-Þ][\w'’.-]*(?:\s+(?:(?:de|du|des|van|von)\s+)?" r"[A-ZÀ-ÖØ-Þ][\w'’.-]*){1,4}"
)
TITLE = re.compile(
    r"(?:direct(?:eur|rice|or)|responsable|assistante?|commerciale?|gérant(?:e)?|"
    r"président(?:e)?|president|CEO|manager|sales manager|account manager|"
    r"sales director|managing director|engineer|ingénieur(?:e)?)"
    r"(?:\s+(?:commerciale?|général(?:e)?|des ventes|technique|administrati[fv]e?))?",
    re.I,
)
COMPANY = re.compile(
    r"(?:(?:SAS|SARL|SA|EURL|Company|Entreprise)\s+[\w '&’.-]{2,60}|"
    r"[\w '&’.-]{2,60}\s+(?:SAS|SARL|Ltd\.?|Inc\.?|LLC))",
    re.I,
)
PHONE = re.compile(
    r"(?:(?:tél(?:éphone)?|tel(?:ephone)?|phone|mobile|fax|portable)\s*[:.]?\s*)?"
    r"\+?[\d ().-]{8,30}",
    re.I,
)
EMAIL = re.compile(r"(?:(?:e-?mail|courriel)\s*:\s*)?[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
WEB = re.compile(
    r"(?:(?:LinkedIn|Facebook|Instagram|Twitter|Site web|Web)\s*:\s*)?"
    r"(?:https?://|www\.)[^\s<>]+",
    re.I,
)
STREET = re.compile(
    r"\d{1,5}\s+(?:bis\s+)?(?:rue|avenue|av\.|boulevard|bd\.|route|"
    r"chemin|allée|place)\s+[\w '’.-]{2,70}",
    re.I,
)
CITY = re.compile(r"\d{5}\s+[A-Za-zÀ-ÿ '’.-]{2,60}(?:\s*[-,]\s*France)?", re.I)
EN_STREET = re.compile(r"\d{1,5}\s+[\w '’.-]{2,60}\s+(?:Street|Road|Avenue|Lane|St\.|Rd\.)", re.I)
ENVIRONMENT = re.compile(
    r"(?:Please consider the environment before printing this (?:e-?mail|message)|"
    r"Pensez à l'environnement avant d'imprimer ce (?:courriel|message|mail))\.?",
    re.I,
)
CONTACT_INTRO = re.compile(
    r"\b(?:contact\w*|joindre|appel\w*|call|write|send|envoy\w*|adresse|address|"
    r"destinataire|recipient|interlocuteur|fournisseur|supplier)\b",
    re.I,
)


@dataclass(frozen=True)
class PreparationOptions:
    max_chars: int = 1600
    overlap: int = 160

    def __post_init__(self) -> None:
        if not 128 <= self.max_chars <= 100000 or not 0 <= self.overlap < self.max_chars // 2:
            raise ValueError("Use max_chars 128..100000 and overlap below half the maximum")

    @property
    def key(self) -> str:
        return f"{PREPARATION_VERSION}/{PROCESSING_VERSION}/{self.max_chars}/{self.overlap}"


@dataclass(frozen=True)
class PreparedChunk:
    section_index: int
    start: int
    end: int
    text: str
    hard_split: bool
    transformations: tuple[str, ...]


@dataclass(frozen=True)
class PreparedEmail:
    sections: tuple[dict[str, Any], ...]
    chunks: tuple[PreparedChunk, ...]


def input_fingerprint(record: dict[str, Any]) -> str:
    value = {
        key: record[key]
        for key in ("original_sha256", "normalized_text", "processed_text", "sections_json")
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def unquote(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        remainder = text[match.end() :]
        if re.match(r"[=+\-]?\d", remainder):
            return match[0]
        return ""

    return re.sub(r"(?m)^[ \t]*(?:>[ \t]*)+", replace, text)


def _contact_kind(value: str) -> str | None:
    if BUSINESS.search(value):
        return None
    # Contacts contain numbers too; accept only full, tightly specified formats first.
    for kind, pattern in (
        ("phone", PHONE),
        ("email", EMAIL),
        ("web", WEB),
        ("address", STREET),
        ("address", CITY),
        ("address", EN_STREET),
    ):
        if pattern.fullmatch(value):
            if kind == "phone":
                if len(re.sub(r"\D", "", value)) not in range(9, 16) or value.isdigit():
                    continue
            if kind == "web" and (VALUABLE.search(value) or "?" in value or "#" in value):
                continue
            return kind
    if VALUABLE.search(value) or any(char.isdigit() for char in value):
        return None
    for kind, pattern in (("title", TITLE), ("company", COMPANY), ("name", NAME)):
        if pattern.fullmatch(value):
            return kind
    return None


def _signature_rule(values: list[str], anchored: bool) -> bool:
    nonempty = [value for value in values if value]
    if not nonempty or len(nonempty) > 16:
        return False
    kinds = [_contact_kind(value) for value in nonempty]
    if None in kinds or "name" not in kinds or not {"phone", "email"}.intersection(kinds):
        return False
    return anchored or (
        {"phone", "email"}.issubset(kinds) and bool({"title", "company"}.intersection(kinds))
    )


def _line_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = record["normalized_text"]
    sections = json.loads(record["sections_json"])
    position = 0
    for section in sections:
        if (
            section["start"] != position
            or section["end"] < position
            or text[position : section["end"]] != section["text"]
        ):
            raise ValueError("Inconsistent processed sections")
        position = section["end"]
    if position != len(text):
        raise ValueError("Incomplete processed sections")
    lines: list[dict[str, Any]] = []
    for source_index, section in enumerate(sections):
        position = section["start"]
        for raw in section["text"].splitlines(keepends=True):
            prefix = re.match(r"^[ \t]*(?:>[ \t]*)+", raw)
            lines.append(
                {
                    "start": position,
                    "end": position + len(raw),
                    "value": (raw[prefix.end() :] if prefix else raw).strip(),
                    "source_section": source_index,
                    "depth": prefix[0].count(">") if prefix else 0,
                    "kind": section["kind"],
                    "excluded": False,
                    "rule": (
                        "upstream_exclusion_reconsidered" if not section["included"] else "retained"
                    ),
                    "confidence": section["confidence"],
                }
            )
            position += len(raw)
    message = 0
    header = False
    previous_depth = 0
    for index, line in enumerate(lines):
        value = line["value"]
        fields = {
            match[1].lower()
            for entry in lines[index : index + 8]
            if (match := HEADER.match(entry["value"]))
        }
        outlook = (
            bool(re.match(r"^(?:De|From)\s*:", value, re.I))
            and bool(fields & {"envoyé", "sent", "date"})
            and bool(fields & {"objet", "subject"})
        )
        marker = bool(REPLY.fullmatch(value) or FORWARD.fullmatch(value) or outlook)
        if marker or (value and line["depth"] != previous_depth):
            if not (marker and header):
                message += 1
            header = marker
        if header and not (marker or HEADER.match(value) or not value):
            header = False
        line["message_index"] = message
        line["header"] = header
        if header:
            line["kind"] = "quoted_header"
        previous_depth = line["depth"] if value else previous_depth
    return lines


def _classify(lines: list[dict[str, Any]]) -> None:
    for _, group in groupby(lines, key=lambda line: (line["message_index"], line["header"])):
        block = list(group)
        if block[0]["header"]:
            continue
        values = [line["value"] for line in block]
        # Footer controls are whole-line rules, including inside quoted messages.
        for line in block:
            if BOILERPLATE.fullmatch(line["value"]) or ENVIRONMENT.fullmatch(line["value"]):
                line.update(
                    excluded=True, rule="standalone_footer", confidence="high", kind="boilerplate"
                )
        for index, line in enumerate(block):
            if MOBILE.fullmatch(line["value"]) and not any(values[index + 1 :]):
                line.update(
                    excluded=True, rule="mobile_footer", confidence="high", kind="signature"
                )
        for index, value in enumerate(values):
            anchored = bool(SIGNOFF.fullmatch(value) or value == "--")
            if not anchored and not NAME.fullmatch(value):
                continue
            if not anchored:
                preceding = next((entry for entry in reversed(values[:index]) if entry), "")
                if not preceding or values[index - 1] or CONTACT_INTRO.search(preceding):
                    continue
            tail = [
                entry["value"] for entry in block[index + int(anchored) :] if not entry["excluded"]
            ]
            if _signature_rule(tail, anchored):
                for line in block[index:]:
                    if not line["excluded"]:
                        line.update(
                            excluded=True,
                            rule="business_signature",
                            confidence="high",
                            kind="signature",
                        )
                break
            if anchored:
                block[index].update(rule="ambiguous_signature_retained", confidence="low")


def _split(
    text: str, start: int, end: int, options: PreparationOptions, section_index: int
) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + options.max_chars, end)
        hard = False
        if stop < end:
            window = text[cursor:stop]
            boundaries = [
                match.end()
                for match in re.finditer(r"\n\s*\n|[.!?]\s+|\n|\s+", window)
                if match.end() >= options.max_chars // 2
            ]
            if boundaries:
                # Paragraphs/sentences get priority over a late arbitrary space.
                semantic = [
                    match.end()
                    for match in re.finditer(r"\n\s*\n|[.!?]\s+", window)
                    if match.end() >= options.max_chars // 2
                ]
                stop = cursor + (semantic[-1] if semantic else boundaries[-1])
            else:
                hard = True
        raw = text[cursor:stop]
        unquoted = unquote(raw)
        transformations: set[str] = set()
        if raw != unquoted:
            transformations.add("quote_prefix_removed")
        if unquoted != unquoted.strip():
            transformations.add("outer_whitespace_trimmed")
        chunk_text = _clean_urls(unquoted.strip(), transformations)
        if chunk_text:
            chunks.append(
                PreparedChunk(
                    section_index, cursor, stop, chunk_text, hard, tuple(sorted(transformations))
                )
            )
        if stop == end:
            break
        next_cursor = stop
        if options.overlap:
            overlap_start = max(cursor + 1, stop - options.overlap)
            boundary = re.search(r"\s+", text[overlap_start:stop])
            if boundary:
                next_cursor = overlap_start + boundary.end()
        cursor = next_cursor
    return chunks


def prepare_email(record: dict[str, Any], options: PreparationOptions) -> PreparedEmail:
    lines = _line_records(record)
    _classify(lines)
    sections: list[dict[str, Any]] = []
    chunks: list[PreparedChunk] = []
    fields = ("source_section", "message_index", "depth", "kind", "excluded", "rule", "confidence")
    for key, group in groupby(lines, key=lambda line: tuple(line[field] for field in fields)):
        entries = list(group)
        section = dict(zip(fields, key))
        section.update(start=entries[0]["start"], end=entries[-1]["end"])
        sections.append(section)
        if not section["excluded"]:
            chunks.extend(
                _split(
                    record["normalized_text"],
                    section["start"],
                    section["end"],
                    options,
                    len(sections) - 1,
                )
            )
    return PreparedEmail(tuple(sections), tuple(chunks))


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def near_features(text: str) -> tuple[list[str], str, str]:
    tokens = TOKEN.findall(text.casefold())
    return tokens, content_id(" ".join(tokens[:3])), content_id(" ".join(tokens[-3:]))


def near_decision(left: str, right: str) -> dict[str, Any] | None:
    a, _, _ = near_features(left)
    b, _, _ = near_features(right)
    if min(len(a), len(b)) < 12:
        return None
    # Protected values must agree even for advisory matches; near matches never share text.
    protected_a = {token for token in a if any(char.isdigit() for char in token) or "@" in token}
    protected_b = {token for token in b if any(char.isdigit() for char in token) or "@" in token}
    if protected_a != protected_b:
        return None
    if set(re.findall(r"https?://\S+", left)) != set(re.findall(r"https?://\S+", right)):
        return None
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    if ratio < 0.96:
        return None
    return {"rule": "token_sequence_ratio", "score": round(ratio, 6), "action": "retain_separate"}


def prepare_emails(
    store: EmailPreparationStore,
    *,
    options: PreparationOptions,
    limit: int = 100,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("Preparation limit must be positive")
    counts = {"attempted": 0, "prepared": 0, "failed": 0}
    status = "complete"
    try:
        for record in islice(store.preparation_candidates(options), limit):
            try:
                result = prepare_email(record, options)
            except Exception:
                store.save_preparation(record, options, None)
                counts["failed"] += 1
            else:
                store.save_preparation(record, options, result)
                counts["prepared"] += 1
            counts["attempted"] += 1
            if progress and counts["attempted"] % 25 == 0:
                progress(
                    f"Attempted {counts['attempted']}; prepared {counts['prepared']}; "
                    f"failed {counts['failed']}"
                )
        if counts["failed"]:
            status = "partial"
        elif counts["attempted"] == limit:
            status = "limited"
    except KeyboardInterrupt:
        status = "interrupted"
    if progress:
        progress(
            f"Attempted {counts['attempted']}; prepared {counts['prepared']}; "
            f"failed {counts['failed']}"
        )
    return {"status": status, "generation_key": options.key, "options": asdict(options), **counts}
