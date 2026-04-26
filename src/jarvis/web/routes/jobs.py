from flask import Blueprint, abort, current_app, render_template, request

bp = Blueprint("jobs", __name__)


@bp.route("/jobs")
def jobs_list():
    store = current_app.config["GET_STORE"]()
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    jobs = store.list_jobs(limit=limit, offset=offset)
    return render_template("jobs_list.html", jobs=jobs, limit=limit, offset=offset)


@bp.route("/jobs/<job_id>")
def job_detail(job_id: str):
    store = current_app.config["GET_STORE"]()
    job = store.get_job(job_id)
    if job is None:
        abort(404)
    return render_template("job_detail.html", job=job)
