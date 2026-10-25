"""Synthetic signature, provenance, boundaries and conservative duplicate rules."""

import json

import pytest

from jarvis.email_cleaning import clean_email
from jarvis.email_preparation import (
    PreparationOptions,
    input_fingerprint,
    near_decision,
    prepare_email,
    unquote,
)


def processed_record(body: str) -> dict[str, str]:
    result = clean_email(body)
    return {
        "email_id": "synthetic",
        "original_sha256": "synthetic",
        "normalized_text": result.normalized_text,
        "processed_text": result.processed_text,
        "sections_json": json.dumps(result.sections_data()),
    }


def texts(body: str) -> str:
    return "\n".join(
        chunk.text for chunk in prepare_email(processed_record(body), PreparationOptions()).chunks
    )


@pytest.mark.parametrize(
    "signature",
    [
        "Cordialement,\nJean Dupont\nDirecteur commercial\nEntreprise Exemple\n"
        "Tél : +33 (0)1 23 45 67 89\njean@example.test\n12 rue des Lilas\n75001 Paris\n"
        "LinkedIn : https://linkedin.com/company/example",
        "Best regards,\nJane Smith\nSales Manager\nExample Ltd\n"
        "Phone: +44 20 1234 5678\njane@example.test\n12 Example Street",
        "Jean Dupont\nResponsable commercial\nTél : 01 23 45 67 89\njean@example.test",
    ],
)
def test_confident_multisignal_signatures_excluded_and_recoverable(signature: str) -> None:
    body = "Merci de livrer AB-123 : 450,00 € avant le 25/10/2026.\n\n" + signature
    record = processed_record(body)
    result = prepare_email(record, PreparationOptions())
    output = "\n".join(chunk.text for chunk in result.chunks)
    assert "AB-123" in output and "450,00 €" in output
    assert "example.test" not in output
    excluded = [section for section in result.sections if section["excluded"]]
    assert any(section["rule"] == "business_signature" for section in excluded)
    assert (
        "".join(record["normalized_text"][s["start"] : s["end"]] for s in result.sections)
        == record["normalized_text"]
    )
    assert sum(s["end"] - s["start"] for s in result.sections) == len(record["normalized_text"])


@pytest.mark.parametrize(
    "fact",
    [
        "Commande AB-123 : 450 €",
        "Livraison le 25/10/2026",
        "Please confirm order 1789",
        "Merci de rappeler Jean",
        "Budget approuvé",
        "Quantité : 12 palettes",
        "Réf XY-001",
        "Une phrase inconnue à préserver.",
        "New facts are available on request.",
        "Confirmer le prix auprès de jane@example.test",
        "Projet Approuvé",
    ],
)
def test_business_or_unknown_text_prevents_signature_block_exclusion(fact: str) -> None:
    body = "Bonjour\nCordialement\nJean Dupont\nDirecteur commercial\n"
    body += "Tél: 01 23 45 67 89\njean@example.test\n" + fact
    output = texts(body)
    assert fact in output
    assert "jean@example.test" in output


@pytest.mark.parametrize("prefix", ["", "> ", ">> "])
def test_signatures_in_multiple_forwarded_and_quoted_messages(prefix: str) -> None:
    historical = (
        "De : Jane Smith\nEnvoyé : lundi\nObjet : Livraison AB-123\n\n"
        "Livraison AB-123 maintenue à 450 €.\nCordialement\nJean Dupont\n"
        "Directeur commercial\njean@example.test\nTél: 01 23 45 67 89\n"
    )
    body = "Nouvelle instruction CD-456.\n-----Message d'origine-----\n"
    body += "\n".join(prefix + line for line in historical.splitlines())
    body += "\nOn Tuesday Jane wrote:\nOld unique instruction EF-789\n"
    body += "Best regards\nJane Smith\nSales Manager\njane@example.test\nPhone: +44 20 1234 5678"
    result = prepare_email(processed_record(body), PreparationOptions())
    output = "\n".join(chunk.text for chunk in result.chunks)
    assert all(identifier in output for identifier in ("AB-123", "CD-456", "EF-789", "450 €"))
    assert "jean@example.test" not in output and "jane@example.test" not in output
    assert len({section["message_index"] for section in result.sections}) >= 3
    assert any(section["kind"] == "quoted_header" for section in result.sections)


def test_newsletter_and_environment_controls_inside_quotes_preserve_offer() -> None:
    body = (
        "On Monday Jane wrote:\nOffre AB-123 à 99 € : commander avant vendredi.\n"
        "Se désabonner : https://example.test/unsubscribe\n"
        "Pensez à l'environnement avant d'imprimer ce message.\n"
    )
    output = texts(body)
    assert "99 €" in output and "AB-123" in output
    assert "unsubscribe" not in output and "environnement" not in output


def test_signature_like_name_without_sufficient_evidence_remains() -> None:
    assert "Jean Dupont" in texts("Information\nCordialement\nJean Dupont")
    assert "jean@example.test" in texts("Jean Dupont\njean@example.test")


def test_short_message_is_one_chunk_and_important_identifiers_are_unchanged() -> None:
    body = "Merci de confirmer PO-2026-456 et https://example.test/p?id=123 pour 123,45 EUR."
    result = prepare_email(processed_record(body), PreparationOptions())
    assert len(result.chunks) == 1 and result.chunks[0].text == body


def test_long_chunks_are_bounded_overlapping_and_cover_every_source_character() -> None:
    body = "\n\n".join("Commande AB-123 : " + "texte utile " * 40 for _ in range(7))
    record = processed_record(body)
    result = prepare_email(record, PreparationOptions(256, 40))
    assert len(result.chunks) > 10
    covered = set()
    overlap_seen = False
    previous_end = 0
    for chunk in result.chunks:
        assert len(chunk.text) <= 256
        assert chunk.end - chunk.start <= 256
        section = result.sections[chunk.section_index]
        assert section["start"] <= chunk.start < chunk.end <= section["end"]
        assert chunk.text == unquote(record["normalized_text"][chunk.start : chunk.end]).strip()
        assert chunk.start >= previous_end - 40
        overlap_seen |= chunk.start < previous_end
        covered.update(range(chunk.start, chunk.end))
        previous_end = chunk.end
    assert overlap_seen
    assert covered == set(range(len(record["normalized_text"])))
    assert all(not chunk.hard_split for chunk in result.chunks)


def test_no_chunk_crosses_message_source_section_or_excluded_signature() -> None:
    body = (
        "New AB-123\nRegards\nJane Smith\nSales Manager\njane@example.test\n"
        "On Monday Jane wrote:\n> Old CD-456\n> Unique history retained"
    )
    record = processed_record(body)
    result = prepare_email(record, PreparationOptions(128, 10))
    upstream = json.loads(record["sections_json"])
    for chunk in result.chunks:
        section = result.sections[chunk.section_index]
        source = upstream[section["source_section"]]
        assert not section["excluded"]
        assert source["start"] <= chunk.start < chunk.end <= source["end"]
    assert "Old CD-456" in "\n".join(chunk.text for chunk in result.chunks)


def test_oversized_unbroken_text_is_recoverable_and_explicitly_flagged() -> None:
    record = processed_record("x" * 450)
    result = prepare_email(record, PreparationOptions(128, 0))
    assert all(len(chunk.text) <= 128 for chunk in result.chunks)
    assert result.chunks[0].hard_split
    assert "".join(chunk.text for chunk in result.chunks) == record["normalized_text"]


def test_empty_and_deterministic_preparation_and_input_validation() -> None:
    assert prepare_email(processed_record(""), PreparationOptions()).chunks == ()
    record = processed_record("A sufficiently small message")
    assert prepare_email(record, PreparationOptions()) == prepare_email(
        record, PreparationOptions()
    )
    assert input_fingerprint(record) == input_fingerprint(dict(reversed(list(record.items()))))
    record["sections_json"] = "[]"
    with pytest.raises(ValueError, match="Incomplete"):
        prepare_email(record, PreparationOptions())


@pytest.mark.parametrize("max_chars,overlap", [(127, 0), (100001, 0), (1600, -1), (128, 64)])
def test_invalid_configuration_is_refused(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        PreparationOptions(max_chars, overlap)


def test_near_duplicates_are_advisory_and_numbers_and_unique_facts_remain_distinct() -> None:
    body = (
        "Please confirm that the requested delivery for order AB-123 will arrive on Monday morning."
    )
    reflow = body.replace("the requested", "the  requested").upper()
    decision = near_decision(body, reflow)
    assert decision and decision["action"] == "retain_separate" and decision["score"] == 1
    assert near_decision(body, body.replace("AB-123", "AB-124")) is None
    assert near_decision(body, body.replace("Monday", "Friday")) is None
    assert near_decision("OK", "Ok") is None


def test_near_duplicate_small_edit_in_long_message_is_retained_separately() -> None:
    body = "We discussed the delivery arrangements with the entire team this morning and agreed "
    body += (
        "that all previously requested documentation should be available for the meeting next week."
    )
    decision = near_decision(body, body.replace("entire", "whole"))
    assert decision and decision["score"] >= 0.96 and decision["action"] == "retain_separate"


def test_upstream_excluded_ambiguous_business_text_is_reinstated() -> None:
    record = processed_record("Hello\n-- \nCommande Reçue")
    assert "Commande Reçue" not in record["processed_text"]
    result = prepare_email(record, PreparationOptions())
    assert "Commande Reçue" in "\n".join(chunk.text for chunk in result.chunks)


@pytest.mark.parametrize("fact", ["> 100 units needed", ">=250 EUR", "> -10 degrees"])
def test_numeric_comparison_operators_are_not_removed_as_quote_prefixes(fact: str) -> None:
    assert fact in texts(fact)


def test_tracking_cleanup_retains_signed_and_business_url_parameters_with_audit_flags() -> None:
    record = processed_record(
        "See https://example.test/product?id=123&utm_source=mail\n"
        "Signed https://example.test/file?token=private&utm_source=keep"
    )
    result = prepare_email(record, PreparationOptions())
    output = "\n".join(chunk.text for chunk in result.chunks)
    assert "product?id=123" in output and "utm_source=mail" not in output
    assert "token=private&utm_source=keep" in output
    assert any("tracking_parameters_removed" in chunk.transformations for chunk in result.chunks)
    assert any("signed_url_retained" in chunk.transformations for chunk in result.chunks)


@pytest.mark.parametrize(
    "value",
    ["1234567890", "https://example.test/PO-123", "https://example.test/file?token=private"],
)
def test_ambiguous_numeric_identifier_and_business_links_prevent_exclusion(value: str) -> None:
    body = "Regards\nJane Smith\nSales Manager\njane@example.test\n" + value
    assert value in texts(body)


def test_quoted_postal_signature_and_mobile_footer_are_detected() -> None:
    body = (
        "On Monday Jane wrote:\n> Regards\n> Jane Smith\n> Sales Manager\n"
        "> jane@example.test\n> 12 Example Street\n> Sent from my iPhone"
    )
    result = prepare_email(processed_record(body), PreparationOptions())
    assert all("jane@example.test" not in chunk.text for chunk in result.chunks)
    assert {section["rule"] for section in result.sections} >= {
        "business_signature",
        "mobile_footer",
    }


@pytest.mark.parametrize("intro", ["Please contact:", "Vous pouvez joindre :", "Send to:", ""])
def test_contact_cards_and_business_recipients_are_not_unmarked_signatures(intro: str) -> None:
    card = "Jane Smith\nSales Manager\njane@example.test\nPhone: +44 20 1234 5678"
    body = intro + "\n\n" + card if intro else card
    output = texts(body)
    assert "jane@example.test" in output and "Jane Smith" in output


def test_unmarked_block_without_blank_boundary_or_two_contacts_remains() -> None:
    body = "Ask Jane Smith:\nJane Smith\nSales Manager\njane@example.test"
    assert "jane@example.test" in texts(body)
    body = "Details below\n\nJane Smith\nSales Manager\njane@example.test"
    assert "jane@example.test" in texts(body)
