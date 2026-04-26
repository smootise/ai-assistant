from flask import Blueprint, current_app, render_template, request

from jarvis.web.services import run_search

bp = Blueprint("search", __name__)

_DEFAULT_TOP_K = 10
_MAX_TOP_K = 50


@bp.route("/search")
def search():
    query = request.args.get("query", "").strip()
    try:
        top_k = max(1, min(_MAX_TOP_K, int(request.args.get("top_k", _DEFAULT_TOP_K))))
    except (ValueError, TypeError):
        top_k = _DEFAULT_TOP_K

    results = []
    error = None

    if query:
        config = current_app.config["JARVIS_CONFIG"]
        data = run_search(config, query, top_k=top_k)
        results = data["results"]
        error = data["error"]

    return render_template(
        "search.html",
        query=query,
        top_k=top_k,
        results=results,
        error=error,
    )
