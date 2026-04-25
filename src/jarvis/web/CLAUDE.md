# src/jarvis/web/CLAUDE.md — Web UI Context

Loaded automatically when working on the web layer. Covers architecture, constraints, and
design decisions for the JARVIS read-only operator console.

---

## V1 Scope

V1 is **read-only only**. No pipeline invocations, no writes to SQLite, no Qdrant reads.
Future prompts will add: uploads, job launching, retrieve/answer endpoints.

---

## Module Map

| Module | Responsibility |
|---|---|
| `app.py` | Flask app factory. Registers blueprints, injects store factory + config. |
| `services.py` | Service layer. Each function accepts `SummaryStore` + args, returns plain dicts for templates. Never raises on missing data — normalizes to `None`/`[]`. |
| `file_preview.py` | ID-first whitelisted file reader. Resolves paths from SQLite entity metadata, never from user input. |
| `routes/` | One blueprint per entity. Routes orchestrate: call service, pass to template. No business logic in routes. |
| `templates/` | Jinja2 server-rendered HTML. Purely presentational — no logic beyond conditionals and loops. |
| `static/` | Single CSS file. No JS framework, no build step. |

---

## Data Access Rules

- **Always go through `SummaryStore` public methods.** Never import `_connect()` or raw SQL from routes or services.
- New read queries belong in `store.py` as public `SummaryStore` methods (not in routes/services).
- Qdrant is never queried by the web layer in V1.

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
