"""Flask application factory for the JARVIS web UI."""

import json
from pathlib import Path
from typing import Any, Dict

from flask import Flask

from jarvis.config import load_config
from jarvis.store import SummaryStore


def create_app(config: Dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask app.

    Args:
        config: pre-loaded JARVIS config dict; if None, load_config() is called.
    """
    if config is None:
        config = load_config()

    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    app.secret_key = config.get("secret_key", "jarvis-dev-key-change-in-prod")

    db_path = config["db_path"]
    repo_root = config.get("repo_root") or Path(__file__).parent.parent.parent.parent
    output_root = Path(repo_root) / config.get("output_root", "OUTPUTS")
    inbox_root = Path(repo_root) / "inbox"

    inbox_raw_dir = inbox_root / "ai_chat" / "chatgpt" / "raw"
    inbox_raw_dir.mkdir(parents=True, exist_ok=True)

    app.config["JARVIS_CONFIG"] = config
    app.config["JARVIS_DB_PATH"] = db_path
    app.config["JARVIS_REPO_ROOT"] = Path(repo_root)
    app.config["JARVIS_OUTPUT_ROOT"] = output_root
    app.config["JARVIS_INBOX_ROOT"] = inbox_root
    app.config["JARVIS_INBOX_RAW_DIR"] = inbox_raw_dir
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    # Jinja filters
    @app.template_filter("pretty_json")
    def pretty_json_filter(value: Any) -> str:
        try:
            if isinstance(value, str):
                value = json.loads(value)
            return json.dumps(value, indent=2, ensure_ascii=False)
        except Exception:
            return str(value)

    @app.template_filter("short_id")
    def short_id_filter(value: str) -> str:
        if not value:
            return "—"
        return value[:12] + ("…" if len(value) > 12 else "")

    # Store factory — routes call this to get a fresh store per request
    def get_store() -> SummaryStore:
        return SummaryStore(db_path)

    app.config["GET_STORE"] = get_store

    # Register blueprints
    from jarvis.web.routes.dashboard import bp as dashboard_bp
    from jarvis.web.routes.sources import bp as sources_bp
    from jarvis.web.routes.conversations import bp as conversations_bp
    from jarvis.web.routes.segments import bp as segments_bp
    from jarvis.web.routes.extracts import bp as extracts_bp
    from jarvis.web.routes.fragments import bp as fragments_bp
    from jarvis.web.routes.uploads import bp as uploads_bp
    from jarvis.web.routes.jobs import bp as jobs_bp
    from jarvis.web.routes.pipeline_jobs import bp as pipeline_jobs_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(sources_bp)
    app.register_blueprint(conversations_bp)
    app.register_blueprint(segments_bp)
    app.register_blueprint(extracts_bp)
    app.register_blueprint(fragments_bp)
    app.register_blueprint(uploads_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(pipeline_jobs_bp)

    @app.errorhandler(404)
    def not_found(e: Any):
        from flask import render_template
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(e: Any):
        from flask import flash, render_template
        flash("File exceeds the 50 MB size limit.", "error")
        return render_template("upload.html"), 413

    return app
