"""Synthetic French/English messages and conservative preservation regressions."""

import pytest

from jarvis.email_cleaning import clean_email


@pytest.mark.parametrize(
    "body",
    [
        "Bonjour,\r\n\r\nLe produit AB-123 coûte 1 299,90 € au 25/10/2026.\r\nMerci.",
        "Order XY_789: $59.95 on 2026-10-25. https://example.invalid/item?id=42&price=59.95",
        "- First item\n  - Nested item\n- Second item\n\nProduct  Price\nAB-123   12.50 €",
    ],
)
def test_plain_content_numbers_lists_tables_and_determinism(body: str) -> None:
    result = clean_email(body)
    assert result.processed_text == body.replace("\r\n", "\n")
    assert result == clean_email(body)
    assert not result.needs_review
    assert all(section.included for section in result.sections)


def test_whitespace_preserves_relative_indentation_and_table_columns() -> None:
    body = "\r\n    Bonjour  \r\n\r\n\r\n    - Article\r\n      - Sous-article\r\n    A  B\r\n"
    result = clean_email(body)
    assert result.processed_text == "Bonjour\n\n- Article\n  - Sous-article\nA  B"
    assert clean_email(result.processed_text).processed_text == result.processed_text


def test_meaningful_indentation_on_first_line_is_retained() -> None:
    body = "  - Nested carryover\n- Top level\n    A  B\n    1  2"
    assert clean_email(body).processed_text == body


def test_nonstandard_signature_separator_is_retained_for_review() -> None:
    body = "Hello\n--\nName"
    result = clean_email(body)
    assert result.processed_text == body
    assert result.needs_review


@pytest.mark.parametrize(
    "header",
    [
        "De : Marie\nEnvoyé : lundi 1 janvier 2024\nÀ : Paul\nObjet : Devis",
        "From: Mary\nSent: Monday, January 1, 2024\nTo: Paul\nSubject: Quote",
        "Le 1 janvier 2024 à 10:00, Marie a écrit :",
        "On Mon, Jan 1, 2024 at 10:00 AM Mary wrote:",
        "-----Original Message-----",
        "---------- Forwarded message ---------",
        "-----Message d'origine-----",
        "Début du message réexpédié :",
    ],
)
def test_reply_forward_boundaries_preserve_only_surviving_older_information(header: str) -> None:
    body = f"Nouvelle réponse\n\n{header}\nAncien devis REF-X: 99,50 €\n> Citation\n>> Plus ancien"
    result = clean_email(body)
    assert result.processed_text == body
    assert result.sections[0].kind == "new"
    assert any(
        section.kind == "quoted" and "Ancien devis" in section.text for section in result.sections
    )
    assert all(section.included for section in result.sections)


def test_interleaved_replies_and_multiple_quote_levels_are_retained() -> None:
    body = "New\n> Old A\nInline answer 123\n>> Very old\n> Old B\nFinal answer 456"
    result = clean_email(body)
    assert result.processed_text == body
    assert [section.kind for section in result.sections] == [
        "new",
        "quoted",
        "new",
        "quoted",
        "new",
    ]


def test_only_strong_signature_removed_and_every_normalized_character_recoverable() -> None:
    body = "Devis 2026: 250 €\n\n-- \nMarie Martin\n+33 1 23 45 67 89"
    result = clean_email(body)
    assert result.processed_text == "Devis 2026: 250 €"
    assert result.sections[-1].kind == "signature"
    assert not result.sections[-1].included
    assert "+33 1 23 45 67 89" in result.sections[-1].text
    assert "".join(section.text for section in result.sections) == result.normalized_text
    for section in result.sections:
        assert result.normalized_text[section.start : section.end] == section.text


def test_signature_does_not_swallow_following_quoted_content() -> None:
    result = clean_email("Réponse\n-- \nMarie\nOn Monday, Mary wrote:\nOld price: $100")
    assert "Marie" not in result.processed_text
    assert "Old price: $100" in result.processed_text
    assert result.sections[-1].kind == "quoted"


@pytest.mark.parametrize(
    "valuable",
    [
        "Please pay 120 €",
        "Order AB-123",
        "Delivery 2026-10-25",
        "Total: 12.50 EUR",
        "A long ordinary message. " * 8,
        "Please deliver the order on Monday",
    ],
)
def test_signature_separator_with_valuable_or_prose_content_is_retained(valuable: str) -> None:
    body = f"Bonjour\n-- \n{valuable}"
    result = clean_email(body)
    assert result.processed_text == body.rstrip()
    assert result.needs_review
    assert result.sections[-1].confidence == "low"


def test_signature_then_inline_quote_does_not_remove_later_answer() -> None:
    result = clean_email("Hello\n-- \nName\n> Earlier question\nMy answer: REF-42 costs 50 €")
    assert "Name" not in result.processed_text
    assert "Earlier question" in result.processed_text
    assert "My answer: REF-42 costs 50 €" in result.processed_text


def test_mobile_like_ordinary_sentence_is_retained() -> None:
    body = "Envoyé depuis Outlook le devis AB-123 coûte 150 €"
    assert clean_email(body).processed_text == body


def test_long_message_has_complete_recoverable_sections() -> None:
    body = "Product AB-123: 150 €\n" * 20000
    result = clean_email(body)
    assert result.processed_text == body.rstrip("\n")
    assert len(result.sections) == 1
    assert result.sections[0].text == result.normalized_text


@pytest.mark.parametrize(
    "footer", ["Sent from my iPhone", "Envoyé de mon iPad", "Get Outlook for Android"]
)
def test_mobile_signature_only_at_message_end(footer: str) -> None:
    assert clean_email(f"Message\n{footer}").processed_text == "Message"
    assert footer in clean_email(f"{footer}\nImportant additional content").processed_text
    assert footer in clean_email(f"> {footer}").processed_text


def test_ambiguous_signoff_and_headers_are_retained_and_flagged() -> None:
    body = "Cordialement,\nLe produit coûte 450 €\nDe : ceci est une étiquette"
    result = clean_email(body)
    assert result.processed_text == body
    assert result.needs_review
    assert result.indicators == ("possible_quote_retained", "possible_signature_retained")
    assert any(section.confidence == "low" and section.included for section in result.sections)


def test_newsletter_preserves_offer_destination_and_business_parameters() -> None:
    body = (
        "AB-123 : 49,90 € jusqu'au 31/12/2026\n"
        "https://shop.invalid/item?id=42&utm_source=email&variant=red&gclid=tracking\n"
        "Se désabonner : https://shop.invalid/unsubscribe?token=secret\n"
        "View this email in your browser https://shop.invalid/view?id=42"
    )
    result = clean_email(body)
    assert result.processed_text == (
        "AB-123 : 49,90 € jusqu'au 31/12/2026\n" "https://shop.invalid/item?id=42&variant=red"
    )
    removed = [section for section in result.sections if not section.included]
    assert all(section.kind == "boilerplate" for section in removed)
    assert "token=secret" in "".join(section.text for section in removed)
    assert "utm_source=email" in result.normalized_text


def test_signed_and_malformed_links_are_retained() -> None:
    body = "https://shop.invalid/a?utm_source=email&signature=abc\nhttps://[invalid/a?utm_x=a"
    result = clean_email(body)
    assert result.processed_text == body
    assert result.needs_review
    assert "signed_url_retained" in result.indicators
    assert "malformed_url_retained" in result.indicators


def test_html_artifacts_and_links_preserve_prices_tables_and_content_for_review() -> None:
    result = clean_email(
        "<div>Produit&nbsp;AB-123<br>Prix 10,00 €</div><table><tr><td>A</td><td>B</td>"
        '</tr></table><p><a href="https://shop.invalid/p?id=8&amp;utm_source=x">Acheter</a></p>'
    )
    assert "Produit AB-123" in result.processed_text
    assert "Prix 10,00 €" in result.processed_text
    assert "A\tB" in result.processed_text
    assert "https://shop.invalid/p?id=8" in result.processed_text
    assert result.needs_review
    assert "html_converted_review" in result.indicators


def test_html_image_alternative_text_is_retained() -> None:
    result = clean_email('<div><img alt="REF-456 à 99 €" src="https://shop.invalid/pixel"></div>')
    assert result.processed_text == "REF-456 à 99 €"
    assert result.needs_review


@pytest.mark.parametrize("body", [None, "", " \r\n\r\n "])
def test_empty_bodies_are_successful_but_need_review(body: str | None) -> None:
    result = clean_email(body)
    assert result.processed_text == ""
    assert result.needs_review
    assert "empty_body" in result.indicators


def test_malformed_content_is_retained_and_flagged() -> None:
    body = "Broken \ufffd \x00 <div>still valuable REF-123"
    result = clean_email(body)
    assert result.processed_text == body
    assert result.needs_review
    assert "malformed_text_retained" in result.indicators
