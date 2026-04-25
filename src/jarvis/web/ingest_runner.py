"""Background ingest runner for the web upload flow.

Module-level function (not a closure) so daemon threads have clean state.
Each invocation opens its own SummaryStore connection — safe for threading.
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict

from jarvis.ingest.pipeline import ingest_chatgpt
from jarvis.store import SummaryStore


logger = logging.getLogger(__name__)


def run_ingest_job(job_id: str, raw_path: Path, config: Dict[str, Any]) -> None:
    """Run a chatgpt ingest job and update the job row when done.

    Called from a daemon thread spawned by the upload route. Opens its own
    SQLite connection so it does not share state with the request thread.
    """
    store = SummaryStore(db_path=config["db_path"])
    store.mark_job_running(job_id)
    logger.info(f"Job {job_id}: starting ingest for {raw_path}")

    try:
        repo_root = Path(config.get("repo_root", "."))
        output_dir = Path(repo_root) / "inbox" / "ai_chat" / "chatgpt"
        result = ingest_chatgpt(
            raw_path=raw_path,
            output_dir=output_dir,
            persist=True,
            config=config,
        )
        store.mark_job_succeeded(job_id, result)
        logger.info(f"Job {job_id}: succeeded — {result['conversation_id']}")
    except Exception:
        error_text = traceback.format_exc()
        store.mark_job_failed(job_id, error_text)
        logger.error(f"Job {job_id}: failed\n{error_text}")
