from flask import Blueprint, current_app, render_template

from jarvis.web.services import get_dashboard_data

bp = Blueprint("dashboard", __name__)


@bp.route("/")
def index():
    store = current_app.config["GET_STORE"]()
    data = get_dashboard_data(store)
    return render_template("dashboard.html", **data)
