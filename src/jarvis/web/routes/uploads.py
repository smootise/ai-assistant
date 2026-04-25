import threading
from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from jarvis.web import services
from jarvis.web.ingest_runner import run_ingest_job

bp = Blueprint("uploads", __name__)

_ALLOWED_SOURCE_TYPES = {"chatgpt"}
_ALLOWED_EXTENSIONS = {".json"}


@bp.route("/upload", methods=["GET"])
def upload_form():
    return render_template("upload.html")


@bp.route("/upload", methods=["POST"])
def upload_submit():
    source_type = request.form.get("source_type", "").strip()
    if source_type not in _ALLOWED_SOURCE_TYPES:
        flash(f"Unsupported source type: {source_type!r}. Only 'chatgpt' is supported.", "error")
        return render_template("upload.html"), 400

    file = request.files.get("file")
    if not file or not file.filename:
        flash("No file selected.", "error")
        return render_template("upload.html"), 400

    safe_name = secure_filename(file.filename)
    if not safe_name:
        flash("Invalid filename.", "error")
        return render_template("upload.html"), 400

    ext = Path(safe_name).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        flash(f"Only .json files are accepted for ChatGPT exports. Got: {ext!r}", "error")
        return render_template("upload.html"), 400

    raw_dir: Path = current_app.config["JARVIS_INBOX_RAW_DIR"]
    store = current_app.config["GET_STORE"]()
    config = current_app.config["JARVIS_CONFIG"]

    saved_path, input_metadata, error = services.save_upload(file, safe_name, raw_dir)
    if error:
        flash(error, "error")
        return render_template("upload.html"), 400

    job_id = store.create_job("ingest_chatgpt", input_metadata)

    t = threading.Thread(
        target=run_ingest_job,
        args=(job_id, saved_path, config),
        daemon=True,
    )
    t.start()

    return redirect(url_for("jobs.job_detail", job_id=job_id), code=303)
