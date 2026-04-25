from flask import Blueprint, abort, current_app, render_template, request

from jarvis.web import file_preview
from jarvis.web.services import get_source_detail, get_sources_list

bp = Blueprint("sources", __name__)


@bp.route("/sources")
def sources_list():
    store = current_app.config["GET_STORE"]()
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    sources = get_sources_list(store, limit=limit, offset=offset)
    return render_template("sources_list.html", sources=sources, limit=limit, offset=offset)


@bp.route("/sources/<source_id>")
def source_detail(source_id: str):
    store = current_app.config["GET_STORE"]()
    data = get_source_detail(store, source_id)
    if data is None:
        abort(404)
    return render_template("source_detail.html", **data)


@bp.route("/sources/<source_id>/raw")
def source_raw(source_id: str):
    store = current_app.config["GET_STORE"]()
    output_root = current_app.config["JARVIS_OUTPUT_ROOT"]
    inbox_root = current_app.config["JARVIS_INBOX_ROOT"]
    try:
        content, kind, filename = file_preview.read_for_source(
            store, source_id, output_root, inbox_root
        )
    except file_preview.PreviewError as exc:
        return render_template("file_preview.html", error=str(exc), filename=source_id), 400
    return render_template("file_preview.html", content=content, kind=kind, filename=filename)
