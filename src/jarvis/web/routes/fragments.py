from flask import Blueprint, abort, current_app, render_template

from jarvis.web import file_preview
from jarvis.web.services import get_fragment_detail

bp = Blueprint("fragments", __name__)


@bp.route("/fragments/<fragment_id>")
def fragment_detail(fragment_id: str):
    store = current_app.config["GET_STORE"]()
    data = get_fragment_detail(store, fragment_id)
    if data is None:
        abort(404)
    return render_template("fragment_detail.html", **data)


@bp.route("/fragments/<fragment_id>/raw")
def fragment_raw(fragment_id: str):
    store = current_app.config["GET_STORE"]()
    output_root = current_app.config["JARVIS_OUTPUT_ROOT"]
    try:
        content, kind, filename = file_preview.read_for_fragment(store, fragment_id, output_root)
    except file_preview.PreviewError as exc:
        return render_template("file_preview.html", error=str(exc), filename=fragment_id), 400
    return render_template("file_preview.html", content=content, kind=kind, filename=filename)
