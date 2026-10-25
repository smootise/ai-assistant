"""Per-email checkpoints with bounded batches and safe failure diagnostics."""

from __future__ import annotations

from itertools import islice
from typing import Any, Callable

from jarvis.email_cleaning import clean_email
from jarvis.email_processing_store import EmailProcessingStore


def process_emails(
    store: EmailProcessingStore,
    *,
    limit: int = 100,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("Processing limit must be positive")
    counts = {
        "attempted": 0,
        "processed": 0,
        "failed": 0,
        "manual_review": 0,
        "already_processed": store.completed_count(),
    }
    status = "complete"
    try:
        for email_id, original in islice(store.candidates(force=force), limit):
            try:
                result = clean_email(original)
            except Exception:
                store.save_result(email_id, original, None)
                counts["failed"] += 1
            else:
                store.save_result(email_id, original, result)
                counts["processed"] += 1
                counts["manual_review"] += int(result.needs_review)
            counts["attempted"] += 1
            if progress and counts["attempted"] % 25 == 0:
                progress(
                    f"Attempted {counts['attempted']}; processed {counts['processed']}; "
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
            f"Attempted {counts['attempted']}; processed {counts['processed']}; "
            f"failed {counts['failed']}"
        )
    return {"status": status, **counts}
