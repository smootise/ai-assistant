from flask import Blueprint, current_app, render_template, request

from jarvis.web.services import run_answer

bp = Blueprint("answer", __name__)

_DEFAULT_TOP_K = 10
_MAX_TOP_K = 50
_DEFAULT_TEMPERATURE = 0.3


@bp.route("/answer", methods=["GET", "POST"])
def answer():
    query = ""
    top_k = _DEFAULT_TOP_K
    temperature = _DEFAULT_TEMPERATURE
    result: dict = {}

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        try:
            top_k = max(1, min(_MAX_TOP_K, int(request.form.get("top_k", _DEFAULT_TOP_K))))
        except (ValueError, TypeError):
            top_k = _DEFAULT_TOP_K
        try:
            raw_temp = float(request.form.get("temperature", _DEFAULT_TEMPERATURE))
            temperature = max(0.0, min(1.0, raw_temp))
        except (ValueError, TypeError):
            temperature = _DEFAULT_TEMPERATURE

        if query:
            config = current_app.config["JARVIS_CONFIG"]
            result = run_answer(config, query, top_k=top_k, temperature=temperature)

    return render_template(
        "answer.html",
        query=query,
        top_k=top_k,
        temperature=temperature,
        result=result,
    )
