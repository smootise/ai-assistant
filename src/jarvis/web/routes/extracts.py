from flask import Blueprint, abort, current_app, render_template

from jarvis.web import file_preview
from jarvis.web.services import get_extract_detail

bp = Blueprint("extracts", __name__)


@bp.route("/extracts/<extract_id>")
def extract_detail(extract_id: str):
    store = current_app.config["GET_STORE"]()
    data = get_extract_detail(store, extract_id)
    if data is None:
        abort(404)
    return render_template("extract_detail.html", **data)


@bp.route("/extracts/<extract_id>/raw")
def extract_raw(extract_id: str):
    store = current_app.config["GET_STORE"]()
    output_root = current_app.config["JARVIS_OUTPUT_ROOT"]
    try:
        content, kind, filename = file_preview.read_for_extract(store, extract_id, output_root)
    except file_preview.PreviewError as exc:
        return render_template("file_preview.html", error=str(exc), filename=extract_id), 400
    return render_template("file_preview.html", content=content, kind=kind, filename=filename)
