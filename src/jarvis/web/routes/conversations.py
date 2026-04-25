from flask import Blueprint, abort, current_app, render_template, request

from jarvis.web.services import get_conversation_detail, get_conversations_list

bp = Blueprint("conversations", __name__)


@bp.route("/conversations")
def conversations_list():
    store = current_app.config["GET_STORE"]()
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    conversations = get_conversations_list(store, limit=limit, offset=offset)
    return render_template(
        "conversations_list.html", conversations=conversations, limit=limit, offset=offset
    )


@bp.route("/conversations/<conversation_id>")
def conversation_detail(conversation_id: str):
    store = current_app.config["GET_STORE"]()
    data = get_conversation_detail(store, conversation_id)
    if data is None:
        abort(404)
    return render_template("conversation_detail.html", **data)
