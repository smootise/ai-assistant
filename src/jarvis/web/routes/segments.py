from flask import Blueprint, abort, current_app, render_template

from jarvis.web import file_preview
from jarvis.web.services import get_segment_detail

bp = Blueprint("segments", __name__)


@bp.route("/segments/<segment_id>")
def segment_detail(segment_id: str):
    store = current_app.config["GET_STORE"]()
    data = get_segment_detail(store, segment_id)
    if data is None:
        abort(404)
    return render_template("segment_detail.html", **data)


@bp.route("/segments/<segment_id>/raw")
def segment_raw(segment_id: str):
    store = current_app.config["GET_STORE"]()
    inbox_root = current_app.config["JARVIS_INBOX_ROOT"]
    try:
        content, kind, filename = file_preview.read_for_segment(store, segment_id, inbox_root)
    except file_preview.PreviewError as exc:
        return render_template("file_preview.html", error=str(exc), filename=segment_id), 400
    return render_template("file_preview.html", content=content, kind=kind, filename=filename)
