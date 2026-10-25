"""Conservative, deterministic email sections; quoted information stays searchable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
from html.parser import HTMLParser
from itertools import groupby
import re
import textwrap
from urllib.parse import urlsplit, urlunsplit

PROCESSING_VERSION = "email-clean-v1"
KINDS = ("new", "quoted", "signature", "boilerplate")
URL = re.compile(r"https?://[^\s<>\"]+", re.I)
REPLY = re.compile(r"^(?:Le .{3,400} a écrit\s*:|On .{3,400} wrote\s*:)$", re.I)
FORWARD = re.compile(
    r"^(?:-{2,}\s*(?:Original Message|Message d'origine|Forwarded message|"
    r"Message transféré)\s*-{2,}|Begin forwarded message\s*:|"
    r"Début du message réexpédié\s*:)$",
    re.I,
)
HEADER = re.compile(r"^(De|From|Envoyé|Sent|Date|À|A|To|Cc|Objet|Subject)\s*:", re.I)
MOBILE = re.compile(
    r"^(?:Sent from my (?:iPhone|iPad|Android)(?: device)?|"
    r"Envoyé de mon (?:iPhone|iPad)|Envoyé depuis (?:mon |l'application )?"
    r"(?:Android|Outlook|Mail)(?: pour (?:Windows(?: 10| 11)?|Android|iOS))?|"
    r"Get Outlook for (?:iOS|Android))\.?$",
    re.I,
)
SIGNOFF = re.compile(
    r"^(?:Cordialement|Bien cordialement|Bien à vous|Regards|"
    r"Best regards|Kind regards|Sincerely)[,.!]?\s*$",
    re.I,
)
BOILERPLATE = re.compile(
    r"^(?:Unsubscribe|Se désabonner|Me désabonner|Vous désabonner|"
    r"View this email in your browser|Afficher cet e-mail dans votre navigateur)"
    r"\s*[:|—-]?\s*<?https?://[^\s<>]+>?[.]?$",
    re.I,
)
VALUABLE = re.compile(
    r"[€$£]|\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b|\b[A-Z]{2,}[-_]\d[\w-]*\b|"
    r"\b(?:EUR|USD|GBP)\b|\b\d+[,.]\d{2}\b"
)
CONTACT_LINE = re.compile(
    r"(?:[A-ZÀ-ÖØ-Þ][\w'’.-]*(?:\s+(?:(?:de|du|des|van|von)\s+)?"
    r"[A-ZÀ-ÖØ-Þ][\w'’.-]*){0,5}|\+?[\d ().-]{6,30}|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})"
)


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "div", "br", "tr", "blockquote", "li"}:
            self.parts.append("\n" + ("- " if tag == "li" else ""))
        if tag in {"td", "th"}:
            self.parts.append("\t")
        if tag == "a":
            self.links.append(dict(attrs).get("href"))
        if tag == "img" and (alternative := dict(attrs).get("alt")):
            self.parts.append(alternative)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "tr", "blockquote", "li"}:
            self.parts.append("\n")
        if tag == "a" and self.links:
            href = self.links.pop()
            if href and href.lower().startswith(("https://", "http://", "mailto:")):
                self.parts.append(f" <{href}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


@dataclass(frozen=True)
class Section:
    kind: str
    confidence: str
    indicator: str
    start: int
    end: int
    text: str
    included: bool


@dataclass(frozen=True)
class CleanedEmail:
    normalized_text: str
    processed_text: str
    sections: tuple[Section, ...]
    indicators: tuple[str, ...]
    needs_review: bool

    def sections_data(self) -> list[dict[str, str | int | bool]]:
        return [asdict(section) for section in self.sections]


def _normalize(body: str, indicators: set[str]) -> str:
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    if re.search(r"<(?:html|body|div|p|table)\b[^>]*>", text, re.I) and re.search(
        r"</(?:html|body|div|p|table)>", text, re.I
    ):
        parser = _VisibleHTML()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
        indicators.add("html_converted_review")
    # Decode spacing artifacts only: decoding arbitrary entities could alter identifiers or URLs.
    text = re.sub(r"&nbsp;|&#160;|&#x0*a0;", " ", text, flags=re.I)
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    if "\ufffd" in text or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        indicators.add("malformed_text_retained")
    if re.search(r"</?[a-z][^>]*>|&[a-z]+;", text, re.I):
        indicators.add("unresolved_markup_retained")
    lines = [line.rstrip() if line.strip() != "--" else line for line in text.split("\n")]
    text = textwrap.dedent("\n".join(lines)).strip("\n")
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)


def _clean_urls(text: str, indicators: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group().rstrip(".,;!)]}")
        suffix = match.group()[len(raw) :]
        try:
            parts = urlsplit(raw)
            pairs = parts.query.split("&")
            keys = [html.unescape(pair.split("=", 1)[0]).lower() for pair in pairs]
            if any(key in {"sig", "signature", "token", "x-amz-signature"} for key in keys):
                indicators.add("signed_url_retained")
                return match.group()
            retained = [
                pair
                for pair, key in zip(pairs, keys)
                if not key.startswith("utm_") and key not in {"fbclid", "gclid", "mc_cid", "mc_eid"}
            ]
            if retained != pairs:
                indicators.add("tracking_parameters_removed")
                return urlunsplit(parts._replace(query="&".join(retained))) + suffix
        except ValueError:
            indicators.add("malformed_url_retained")
        return match.group()

    return URL.sub(replace, text)


def _quote_header(lines: list[str], index: int) -> bool:
    first = HEADER.match(lines[index].strip())
    if not first or first[1].lower() not in {"de", "from"}:
        return False
    fields = {
        match[1].lower()
        for line in lines[index : index + 8]
        if (match := HEADER.match(line.strip()))
    }
    return bool(fields & {"envoyé", "sent", "date"}) and bool(fields & {"objet", "subject"})


def _signature_is_confident(lines: list[str], index: int) -> bool:
    tail: list[str] = []
    for next_index in range(index + 1, len(lines)):
        value = lines[next_index].strip()
        if (
            value.startswith(">")
            or REPLY.fullmatch(value)
            or FORWARD.fullmatch(value)
            or _quote_header(lines, next_index)
        ):
            break
        tail.append(value)
        if len(tail) > 8:
            return False
    return bool(any(tail)) and all(
        not line
        or (len(line) <= 100 and not VALUABLE.search(line) and CONTACT_LINE.fullmatch(line))
        for line in tail
    )


def clean_email(body: str | None) -> CleanedEmail:
    indicators: set[str] = set()
    normalized = _normalize(body or "", indicators)
    if not normalized.strip():
        indicators.add("empty_body")
    lines = normalized.splitlines(keepends=True)
    labels: list[tuple[str, str, str, bool]] = []
    quoted_tail = False
    signature_tail = False
    signature_confident = False
    for index, line in enumerate(lines):
        value = line.strip()
        if REPLY.fullmatch(value) or FORWARD.fullmatch(value) or _quote_header(lines, index):
            quoted_tail = True
            signature_tail = False
        if quoted_tail or value.startswith(">"):
            signature_tail = False
            labels.append(
                ("quoted", "high", "quote_boundary" if quoted_tail else "quote_prefix", True)
            )
            continue
        if BOILERPLATE.fullmatch(value):
            labels.append(("boilerplate", "high", "standalone_newsletter_control", False))
            continue
        if MOBILE.fullmatch(value) and not any(part.strip() for part in lines[index + 1 :]):
            labels.append(("signature", "high", "mobile_footer", False))
            continue
        if value == "--":
            signature_tail = True
            signature_confident = line.rstrip("\n") == "-- " and _signature_is_confident(
                lines, index
            )
            if not signature_confident:
                indicators.add("possible_signature_retained")
        if signature_tail:
            labels.append(
                (
                    "signature",
                    "high" if signature_confident else "low",
                    "standard_signature_delimiter",
                    not signature_confident,
                )
            )
        elif SIGNOFF.fullmatch(value):
            indicators.add("possible_signature_retained")
            labels.append(("signature", "low", "ambiguous_signoff", True))
        else:
            if HEADER.match(value) or re.match(r"^(?:Le .+ a écrit|On .+ wrote)", value, re.I):
                indicators.add("possible_quote_retained")
            labels.append(("new", "default", "unclassified_retained", True))
    sections: list[Section] = []
    offset = 0
    for (kind, confidence, indicator, included), group in groupby(
        zip(lines, labels), key=lambda item: item[1]
    ):
        text = "".join(line for line, _ in group)
        sections.append(
            Section(kind, confidence, indicator, offset, offset + len(text), text, included)
        )
        offset += len(text)
    processed = "".join(section.text for section in sections if section.included).strip("\n")
    processed = _clean_urls(processed, indicators)
    needs_review = bool(indicators - {"tracking_parameters_removed"})
    return CleanedEmail(
        normalized, processed, tuple(sections), tuple(sorted(indicators)), needs_review
    )
