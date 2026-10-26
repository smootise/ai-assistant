"""Hand-authored synthetic annotations; no detector generates these labels."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from tools.email_signature_benchmark import digest

FR = (
    "Cordialement,\nAlice Martin\nResponsable commerciale\nAtelier Exemple\nLyon\n"
    "Tél : 01 23 45 67 89\nalice@example.invalid"
)
EN = (
    "Best regards,\nBob Smith\nAccount Manager\nExample Workshop\nLondon\n"
    "Phone: +44 20 1234 5678\nbob@example.invalid"
)

# Authored synthetic labels are independent of detector output; no real human review is implied.
CASES = [
    ("fr-business", "Bonjour,\nLa réunion est fixée à mardi.\n\n<S>" + FR + "</S>"),
    ("en-business", "Hello,\nThe meeting is on Tuesday.\n\n<S>" + EN + "</S>"),
    ("fr-short", "Reçu.\n\n<S>Cordialement,\nAlice Martin</S>"),
    ("en-short", "Received.\n\n<S>Regards,\nBob Smith</S>"),
    ("fr-mobile", "Dossier reçu.\n\n<S>Envoyé de mon iPhone</S>"),
    ("en-mobile", "Files received.\n\n<S>Sent from my iPhone</S>"),
    ("standard", "Pièces reçues.\n\n<S>-- \nAlice Martin\nalice@example.invalid</S>"),
    (
        "unmarked",
        "À mardi.\n\n<S>Alice Martin\nResponsable\nAtelier Exemple\n"
        "01 23 45 67 89\nalice@example.invalid</S>",
    ),
    ("social", "Bien reçu.\n\n<S>" + FR + "\nLinkedIn : https://example.invalid/alice</S>"),
    (
        "crlf",
        "Hello,\r\nSee you Tuesday.\r\n\r\n<S>Regards,\r\n"
        "  Bob Smith  \r\n  bob@example.invalid</S>",
    ),
    ("unicode", "Résumé reçu 👋.\n\n<S>Cordialement,\nÉlodie Noël\nélodie@example.invalid</S>"),
    (
        "outlook-fr",
        "Merci.\n\n<S>" + FR + "</S>\n\nDe : Bob <bob@example.invalid>\nEnvoyé : lundi\n"
        "Objet : rendez-vous\n\nÀ mardi.\n\n<S>" + EN + "</S>",
    ),
    (
        "outlook-en",
        "Thanks.\n\nFrom: Alice <alice@example.invalid>\nSent: Monday\n"
        "Subject: meeting\n\nSee you.\n\n<S>" + FR + "</S>",
    ),
    (
        "gmail",
        "Bien reçu.\n\nLe lundi 5 octobre 2026 à 10:00, Bob a écrit :\n> À mardi.\n> \n"
        "<S>> Regards,\n> Bob Smith\n> bob@example.invalid</S>",
    ),
    (
        "nested",
        "OK.\n\nOn Monday, Alice wrote:\n> OK.\n> On Sunday, Bob wrote:\n>> Merci.\n"
        "<S>>> Cordialement,\n>> Alice Martin\n>> alice@example.invalid</S>",
    ),
    (
        "forward",
        "Pour information.\n\n----- Forwarded message -----\nFrom: Bob\n"
        "Subject: meeting\n\nHello.\n\n<S>" + EN + "</S>",
    ),
    (
        "duplicate-signatures",
        "Message.\n<S>"
        + FR
        + "</S>\n\nFrom: Alice\nSent: Monday\nSubject: RE\n\nPrevious.\n<S>"
        + FR
        + "</S>",
    ),
    (
        "postscript",
        "Bonjour.\n\n<S>"
        + FR
        + "</S>\n\n<C>PS : confirmer le devis de 450 EUR avant vendredi.</C>",
    ),
    (
        "order-after-signoff",
        "Reçu.\n\nCordialement,\n<C>Veuillez livrer 25 pièces REF-123 au prix de 12,50 EUR.</C>",
    ),
    (
        "thanks-instruction",
        "Hello.\n\nThanks,\n<C>Please approve invoice INV-2026 for 500 EUR.</C>",
    ),
    (
        "contact-instruction",
        "<C>Pour la livraison, contactez :\nAlice Martin\n01 23 45 67 89\n"
        "alice@example.invalid</C>",
    ),
    (
        "order-form",
        "<C>ORDER FORM\nProduct: REF-123\nQuantity: 50\nPrice: 200 EUR\n"
        "Supplier: alice@example.invalid\nPhone: 01 23 45 67 89</C>",
    ),
    (
        "newsletter",
        "<C>Offre de la semaine : 30 % sur les produits.\nCommander avant vendredi.\n"
        "https://example.invalid/shop</C>\nSe désabonner : https://example.invalid/unsubscribe",
    ),
    ("hyphen-list", "<C>Next steps:\n- Confirm order\n- Pay 500 EUR\n- Deliver Friday</C>"),
    ("price-separator", "<C>Proposal\n--\nOrder total\n500 EUR\nDelivery Friday</C>"),
    ("empty", ""),
    (
        "no-signature",
        "<C>La livraison doit arriver avant le 12/11/2026. Le contrat est validé.</C>",
    ),
    ("only-contact", "<C>alice@example.invalid</C>"),
    ("html-extracted", "Bonjour,\nLa pièce jointe est correcte.\n\n<S>" + FR + "</S>"),
    (
        "long",
        "<C>"
        + "Le dossier contient les informations suivantes. " * 90
        + "</C>\n\n<S>"
        + FR
        + "</S>",
    ),
]


def annotated(identity: str, marked: str) -> dict[str, Any]:
    body = ""
    labels: dict[str, list[list[int]]] = {"S": [], "C": []}
    cursor = 0
    for match in re.finditer(r"<([SC])>(.*?)</\1>", marked, re.S):
        body += marked[cursor : match.start()]
        start = len(body)
        body += match[2]
        labels[match[1]].append([start, len(body)])
        cursor = match.end()
    body += marked[cursor:]
    return {
        "id": identity,
        "body": body,
        "sender": "Alice Martin <alice@example.invalid>",
        "format": "html-extracted" if identity == "html-extracted" else "plain",
        "label": {
            "status": "reviewed",
            "source": "synthetic-authored",
            "reviewer": "synthetic-author-v1",
            "body_sha256": digest(body),
            "signatures": labels["S"],
            "critical": labels["C"],
        },
    }


def fixtures() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = [annotated(identity, marked) for identity, marked in CASES]
    # Same signature but different authored messages; references are unlabeled evidence.
    reference = []
    for index in range(4):
        body = f"Bonjour,\nRendez-vous numéro {index}.\n\n{FR}"
        reference.append(
            {
                "id": f"reference-{index}",
                "body": body,
                "sender": "Alice Martin <alice@example.invalid>",
                "format": "plain",
            }
        )
    return evaluation, reference


def main() -> None:
    evaluation, reference = fixtures()
    root = Path("tests/fixtures")
    for name, records in (
        ("signature_benchmark.json", evaluation),
        ("signature_reference.json", reference),
    ):
        (root / name).write_text(
            json.dumps(records, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
