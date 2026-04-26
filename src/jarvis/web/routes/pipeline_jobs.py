import threading
from typing import Optional, Tuple

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from jarvis.web import services
from jarvis.web.extract_runner import run_extract_job
from jarvis.web.fragment_runner import run_fragment_job

bp = Blueprint("pipeline_jobs", __name__)


def _parse_segment_range(
    from_raw: str, to_raw: str
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Parse and validate from_segment / to_segment form values.

    Returns (from_segment, to_segment, error_message). error_message is None on success.
    """
    from_seg: Optional[int] = None
    to_seg: Optional[int] = None

    if from_raw.strip():
        try:
            from_seg = int(from_raw.strip())
            if from_seg < 0:
                return None, None, "from_segment must be >= 0."
        except ValueError:
            return None, None, "from_segment must be an integer."

    if to_raw.strip():
        try:
            to_seg = int(to_raw.strip())
            if to_seg < 0:
                return None, None, "to_segment must be >= 0."
        except ValueError:
            return None, None, "to_segment must be an integer."

    if from_seg is not None and to_seg is not None and from_seg > to_seg:
        return None, None, f"from_segment ({from_seg}) must be <= to_segment ({to_seg})."

    return from_seg, to_seg, None


# ---------------------------------------------------------------------------
# Extract routes
# ---------------------------------------------------------------------------

@bp.route("/conversations/<conversation_id>/extract", methods=["GET"])
def extract_form(conversation_id: str):
    store = current_app.config["GET_STORE"]()
    detail = services.get_conversation_detail(store, conversation_id)
    if detail is None:
        abort(404)
    return render_template(
        "pipeline_extract_form.html",
        conversation=detail["conversation"],
    )


@bp.route("/conversations/<conversation_id>/extract", methods=["POST"])
def extract_submit(conversation_id: str):
    store = current_app.config["GET_STORE"]()
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        abort(404)

    from_raw = request.form.get("from_segment", "")
    to_raw = request.form.get("to_segment", "")
    force = request.form.get("force") == "on"
    persist = request.form.get("persist") == "on"

    from_seg, to_seg, range_error = _parse_segment_range(from_raw, to_raw)
    if range_error:
        flash(range_error, "error")
        return render_template(
            "pipeline_extract_form.html",
            conversation=conversation,
        ), 400

    input_metadata = {
        "conversation_id": conversation_id,
        "from_segment": from_seg,
        "to_segment": to_seg,
        "force": force,
        "persist": persist,
    }
    options = {
        "from_segment": from_seg if from_seg is not None else 0,
        "to_segment": to_seg,
        "force": force,
        "persist": persist,
    }

    config = current_app.config["JARVIS_CONFIG"]
    job_id = store.create_job("extract_segments", input_metadata)

    t = threading.Thread(
        target=run_extract_job,
        args=(job_id, conversation_id, options, config),
        daemon=True,
    )
    t.start()

    return redirect(url_for("jobs.job_detail", job_id=job_id), code=303)


# ---------------------------------------------------------------------------
# Fragment routes
# ---------------------------------------------------------------------------

@bp.route("/conversations/<conversation_id>/fragment", methods=["GET"])
def fragment_form(conversation_id: str):
    store = current_app.config["GET_STORE"]()
    detail = services.get_conversation_detail(store, conversation_id)
    if detail is None:
        abort(404)
    return render_template(
        "pipeline_fragment_form.html",
        conversation=detail["conversation"],
        has_extracts=detail["has_extracts"],
    )


@bp.route("/conversations/<conversation_id>/fragment", methods=["POST"])
def fragment_submit(conversation_id: str):
    store = current_app.config["GET_STORE"]()
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        abort(404)

    has_extracts = store.count_extracts_for_conversation(conversation_id) > 0
    if not has_extracts:
        flash(
            "No extracts found for this conversation. Run extract-segments first.",
            "error",
        )
        return render_template(
            "pipeline_fragment_form.html",
            conversation=conversation,
            has_extracts=False,
        ), 400

    from_raw = request.form.get("from_segment", "")
    to_raw = request.form.get("to_segment", "")
    force = request.form.get("force") == "on"
    persist = request.form.get("persist") == "on"
    embed = request.form.get("embed") == "on"

    if embed and not persist:
        flash("embed requires persist to be enabled.", "error")
        return render_template(
            "pipeline_fragment_form.html",
            conversation=conversation,
            has_extracts=True,
        ), 400

    from_seg, to_seg, range_error = _parse_segment_range(from_raw, to_raw)
    if range_error:
        flash(range_error, "error")
        return render_template(
            "pipeline_fragment_form.html",
            conversation=conversation,
            has_extracts=True,
        ), 400

    input_metadata = {
        "conversation_id": conversation_id,
        "from_segment": from_seg,
        "to_segment": to_seg,
        "force": force,
        "persist": persist,
        "embed": embed,
    }
    options = {
        "from_segment": from_seg if from_seg is not None else 0,
        "to_segment": to_seg,
        "force": force,
        "persist": persist,
        "embed": embed,
    }

    config = current_app.config["JARVIS_CONFIG"]
    job_id = store.create_job("fragment_extracts", input_metadata)

    t = threading.Thread(
        target=run_fragment_job,
        args=(job_id, conversation_id, options, config),
        daemon=True,
    )
    t.start()

    return redirect(url_for("jobs.job_detail", job_id=job_id), code=303)
