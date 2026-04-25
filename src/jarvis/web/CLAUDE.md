# src/jarvis/web/CLAUDE.md — Web UI Context

Loaded automatically when working on the web layer. Covers architecture, constraints, and
design decisions for the JARVIS read-only operator console.

---

## V1 Scope

V1 includes a **read-only browser console** plus an **upload + ingest flow**. No Qdrant reads.
Future prompts will add: extract/fragment/retrieve/answer jobs from the UI.

---

## Module Map

| Module | Responsibility |
|---|---|
| `app.py` | Flask app factory. Registers blueprints, injects store factory + config. Sets `MAX_CONTENT_LENGTH`, creates `JARVIS_INBOX_RAW_DIR`. |
| `services.py` | Service layer. Each function accepts `SummaryStore` + args, returns plain dicts for templates. Never raises on missing data — normalizes to `None`/`[]`. `save_upload()` is the only place files are written. |
| `file_preview.py` | ID-first whitelisted file reader. Resolves paths from SQLite entity metadata, never from user input. |
| `ingest_runner.py` | Module-level `run_ingest_job()`. Called from a daemon thread; opens its own `SummaryStore` connection; marks job running → succeeded/failed. |
| `routes/` | One blueprint per entity. Routes orchestrate: call service, pass to template. No business logic in routes. |
| `templates/` | Jinja2 server-rendered HTML. Purely presentational — no logic beyond conditionals and loops. |
| `static/` | Single CSS file. No JS framework, no build step. |

---

## Data Access Rules

- **Always go through `SummaryStore` public methods.** Never import `_connect()` or raw SQL from routes or services.
- New read queries belong in `store.py` as public `SummaryStore` methods (not in routes/services).
- Qdrant is never queried by the web layer.

## Upload + Ingest Rules

- Files may be written to disk **only via `services.save_upload()`** into `JARVIS_INBOX_RAW_DIR` (`inbox/ai_chat/chatgpt/raw/`). No other disk writes from the web layer.
- Uploaded filenames are always sanitized with `werkzeug.utils.secure_filename` before use.
- After a file is saved, the route creates a job row (`store.create_job()`), then spawns a daemon thread running `ingest_runner.run_ingest_job()`. Routes never call `ingest_chatgpt` directly.
- Job state mutations (`mark_job_running`, `mark_job_succeeded`, `mark_job_failed`) go through `SummaryStore` public methods only.
- The job status page (`/jobs/<id>`) auto-refreshes via `<meta http-equiv="refresh" content="2">` only while the job is `pending` or `running`.
- In-flight jobs that are `running` when the server shuts down will remain stuck in `running` — this is a known V1 limitation. A startup sweep to fix orphaned running jobs is deferred.

---

## File Preview Rules

- Paths are **always resolved from entity IDs in SQLite**, never from user-supplied path strings.
- Every resolved path is validated against the whitelist: `OUTPUTS/` and `inbox/ai_chat/` under `repo_root`.
- Symlink escapes are rejected by `Path.resolve(strict=True)` + whitelist check.
- Size cap: 2 MB. Allowed extensions: `.json`, `.md`, `.txt`.

---

## Empty-State Contract

Every list route and every detail page must render cleanly with zero or missing data:
- Lists: show a placeholder message (not an empty table or error).
- Detail pages: show metadata; render missing linked records as `—` or an explanatory note.
- Missing entity (bad ID): return 404 via `abort(404)`, rendered by `templates/404.html`.

---

## Template Conventions

- Extend `base.html`; override `{% block breadcrumb %}` and `{% block content %}`.
- Breadcrumbs must reflect the data lineage: Source → Conversation → Segment → Extract → Fragment.
- Use the `short_id` Jinja filter for long IDs: `{{ id | short_id }}`.
- Use the `pretty_json` Jinja filter for JSON blobs in pre blocks.
- `action-link` class for secondary links; `badge` class for status values.
