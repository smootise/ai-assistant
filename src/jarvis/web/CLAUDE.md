# src/jarvis/web/CLAUDE.md — Web UI Context

Loaded automatically when working on the web layer. Covers architecture, constraints, and
design decisions for the JARVIS read-only operator console.

---

## V1 Scope

V1 includes: browse lineage · upload + ingest · extract-segments job · fragment-extracts job · search (retrieve) · answer (RAG with citations) · job tracking. Complete.

---

## Module Map

| Module | Responsibility |
|---|---|
| `app.py` | Flask app factory. Registers blueprints, injects store factory + config. Sets `MAX_CONTENT_LENGTH`, creates `JARVIS_INBOX_RAW_DIR`. |
| `services.py` | Service layer. Each function accepts `SummaryStore` + args, returns plain dicts for templates. Never raises on missing data — normalizes to `None`/`[]`. `save_upload()` is the only place files are written. |
| `file_preview.py` | ID-first whitelisted file reader. Resolves paths from SQLite entity metadata, never from user input. |
| `ingest_runner.py` | Module-level `run_ingest_job()`. Called from a daemon thread; opens its own `SummaryStore` connection; marks job running → succeeded/failed. |
| `extract_runner.py` | Module-level `run_extract_job(job_id, conversation_id, options, config)`. Same thread pattern as ingest. Calls `SegmentExtractor` + `MemoryLayer` directly. |
| `fragment_runner.py` | Module-level `run_fragment_job(job_id, conversation_id, options, config)`. Calls `Fragmenter` + `MemoryLayer` directly. Guards `embed→persist`. |
| `routes/search.py` | `GET /search` — synchronous fragment retrieval. Calls `services.run_search`. |
| `routes/answer.py` | `GET+POST /answer` — synchronous RAG. Calls `services.run_answer`. POST to avoid browser replay. |
| `routes/` | One blueprint per entity + `pipeline_jobs.py` for extract/fragment launch. Routes orchestrate: validate → call service/runner → render or redirect. No business logic in routes. |
| `templates/` | Jinja2 server-rendered HTML. Purely presentational — no logic beyond conditionals and loops. |
| `static/` | Single CSS file. No JS framework, no build step. |

---

## Data Access Rules

- **Always go through `SummaryStore` public methods.** Never import `_connect()` or raw SQL from routes or services.
- New read queries belong in `store.py` as public `SummaryStore` methods (not in routes/services).
- Qdrant is queried only by `services.run_search` and `services.run_answer` via the CLI helpers `run_retrieval` / `generate_answer`. No other route touches Qdrant.

## Retrieve + Answer Rules

- Both routes are **synchronous** — no background jobs. Ollama latency (5–30s) is acceptable for V1.
- `services.run_search` and `services.run_answer` call `jarvis.cli.run_retrieval` and
  `jarvis.cli.generate_answer` directly. Do not duplicate retrieval or answer logic.
- Services swallow all `ConnectionError` / `RuntimeError` / unexpected exceptions and return them as an `"error"` string in the result dict. Routes never need to catch these.
- `min_results=3` and `min_score=0.50` are fixed defaults — not exposed in the V1 form.

## Upload + Ingest Rules

- Files may be written to disk **only via `services.save_upload()`** into `JARVIS_INBOX_RAW_DIR` (`inbox/ai_chat/chatgpt/raw/`). No other disk writes from the web layer.
- Uploaded filenames are always sanitized with `werkzeug.utils.secure_filename` before use.
- After a file is saved, the route creates a job row (`store.create_job()`), then spawns a daemon thread running `ingest_runner.run_ingest_job()`. Routes never call `ingest_chatgpt` directly.
- Job state mutations (`mark_job_running`, `mark_job_succeeded`, `mark_job_failed`) go through `SummaryStore` public methods only.
- The job status page (`/jobs/<id>`) auto-refreshes via `<meta http-equiv="refresh" content="2">` only while the job is `pending` or `running`.
- In-flight jobs that are `running` when the server shuts down will remain stuck in `running` — this is a known V1 limitation. A startup sweep to fix orphaned running jobs is deferred.

## Pipeline Job Rules (extract + fragment)

- `job_type` values: `extract_segments`, `fragment_extracts` (free-form TEXT, no migration needed).
- `input_metadata` shape — extract: `{conversation_id, from_segment, to_segment, force, persist}`; fragment: adds `embed`.
- `result` shape — extract: `{conversation_id, segments_processed, extracts_persisted, extract_ids[], extract_ids_truncated}`; fragment: `{conversation_id, fragments_produced, fragments_persisted, embedded, skipped_segments[], fragment_ids[], fragment_ids_truncated}`. `extract_ids` / `fragment_ids` capped at 200 entries to bound result blob size.
- **`embed` requires `persist`** — enforced at the route level (400 + flash) and as defense-in-depth inside the runner (`ValueError`).
- **Fragment prereq guard** — fragment form page blocks submission (no submit button) when `store.count_extracts_for_conversation()` returns 0; POST also rejected with 400.
- Deterministic IDs in results: `extract_id = f"{segment_id}_x"`, `fragment_id = f"{extract_id}_f{idx:03d}"`. Runners can build these without querying SQLite.
- Force semantics are replicated exactly from `cli.py` — do not soften.

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
