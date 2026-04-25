# JARVIS Web App

## Overview

The web app is an operator console for browsing and ingesting data into JARVIS.
It runs locally at `http://localhost:5000` and requires no external services beyond a populated
SQLite database (`data/jarvis.db`).

V1 scope: browsing the data lineage + uploading ChatGPT export files for ingest.
Not yet supported: extract/fragment/retrieve/answer via the UI.

---

## Starting the server

```bash
python -m jarvis.cli serve                     # default: 127.0.0.1:5000
python -m jarvis.cli serve --port 8080
python -m jarvis.cli serve --debug             # auto-reload + verbose errors
```

The server binds to `127.0.0.1` only. It is not intended for network exposure.

---

## Pages and routes

| Route | Page | Data shown |
|---|---|---|
| `GET /` | Dashboard | Per-table row counts; 10 most-recent records per entity |
| `GET /sources` | Sources list | All source files; paginated |
| `GET /sources/<id>` | Source detail | Metadata + linked conversations |
| `GET /sources/<id>/raw` | File preview | Raw file from `storage_path` (whitelisted) |
| `GET /conversations` | Conversations list | All conversations; paginated |
| `GET /conversations/<id>` | Conversation detail | Metadata + ordered segment list |
| `GET /segments/<id>` | Segment detail | Full `segment_text`; linked extract if present |
| `GET /segments/<id>/raw` | File preview | `inbox/ai_chat/…/segments/segment_NNN.json` |
| `GET /extracts/<id>` | Extract detail | Metadata; ordered statements; linked fragments |
| `GET /extracts/<id>/raw` | File preview | `OUTPUTS/…/extracts/extract_NNN.json` |
| `GET /fragments/<id>` | Fragment detail | `retrieval_text`; linked statements; parent chain |
| `GET /fragments/<id>/raw` | File preview | `OUTPUTS/…/fragments/segment_NNN/fragment_NNN.json` |
| `GET /upload` | Upload form | Source-type selector + file picker |
| `POST /upload` | Upload handler | Saves file, creates job, launches ingest thread → 303 to `/jobs/<id>` |
| `GET /jobs` | Jobs list | All ingest jobs, newest first |
| `GET /jobs/<id>` | Job status | Status, timestamps, result links or error traceback |

Missing linked records (e.g. a segment with no extract yet) render a placeholder message —
pages never 500 on absent data.

---

## Data lineage navigation

Every detail page shows breadcrumbs and links that follow the lineage:

```
Source → Conversation → Segment → Extract → Fragment
```

Each page links up to its parent and down to its children where rows exist in SQLite.

---

## Upload safety

Files are **never read from user-supplied paths**. The upload route:

1. Validates `source_type` is in the server-side allowlist (`chatgpt` only for now).
2. Sanitizes the filename via `werkzeug.utils.secure_filename`.
3. Rejects non-`.json` extensions.
4. Enforces a 50 MB size cap via Flask `MAX_CONTENT_LENGTH` (returns 413).
5. Writes to a fixed destination: `inbox/ai_chat/chatgpt/raw/<UTC-timestamp>_<safe_name>.json`.
6. Never reads back a path from the request.

After saving, a `jobs` row is created and a daemon thread runs the ingest. The status page
(`/jobs/<id>`) auto-refreshes every 2 seconds while the job is pending or running.

---

## File preview

File preview is **ID-first**: paths are resolved from the entity's SQLite row, never from
a user-supplied path string. Each resolved path is validated against a strict whitelist before
reading:

- Approved roots: `OUTPUTS/` and `inbox/ai_chat/` under the repo root.
- Symlink escapes rejected via `Path.resolve(strict=True)` + whitelist check.
- Path traversal (`..`) rejected.
- Size cap: 2 MB.
- Allowed extensions: `.json`, `.md`, `.txt`.
- JSON files are pretty-printed.

There is no `?path=` endpoint — no user-supplied paths are ever read.

---

## Code layout

```
src/jarvis/web/
  __init__.py          exports create_app()
  app.py               Flask factory — registers blueprints, Jinja filters, store factory
  services.py          thin service layer — one function per page, returns plain dicts
  file_preview.py      ID-first whitelisted file reader
  ingest_runner.py     module-level run_ingest_job() called from daemon threads
  routes/
    dashboard.py       GET /
    sources.py         GET /sources, /sources/<id>, /sources/<id>/raw
    conversations.py   GET /conversations, /conversations/<id>
    segments.py        GET /segments/<id>, /segments/<id>/raw
    extracts.py        GET /extracts/<id>, /extracts/<id>/raw
    fragments.py       GET /fragments/<id>, /fragments/<id>/raw
    uploads.py         GET /upload, POST /upload
    jobs.py            GET /jobs, /jobs/<id>
  templates/
    base.html          shared layout (nav, breadcrumb slot, content slot, head_extra slot)
    dashboard.html
    sources_list.html / source_detail.html
    conversations_list.html / conversation_detail.html
    segment_detail.html
    extract_detail.html
    fragment_detail.html
    file_preview.html
    upload.html
    jobs_list.html / job_detail.html
    404.html
  static/
    styles.css         single stylesheet — system fonts, no framework
```

---

## Architecture

The web layer sits above `SummaryStore` and reads SQLite only:

```
Browser
  └── Flask routes (routes/)
        └── services.py  (orchestration; normalizes missing data to None/[])
              └── SummaryStore public methods
                    └── SQLite  data/jarvis.db
```

No Qdrant, no Ollama, no disk writes in V1. The web layer never calls `SummaryStore._connect()`
directly — all queries go through public methods. New read queries belong in `store.py` as
public `SummaryStore` methods, not in routes or services.

---

## Extending for V2

Planned additions (not yet implemented):

- **Uploads**: POST endpoint + ingest job trigger; new blueprint `routes/ingest.py`.
- **Job status**: polling endpoint for long-running pipeline jobs.
- **Retrieve/answer**: query form + citation drill-down using existing `retrieve` and `answer` CLI logic.

To add a new page:
1. Add any new read queries to `SummaryStore` in `store.py`.
2. Add a service function in `services.py`.
3. Add a route in the relevant blueprint (or a new one in `routes/`).
4. Add a Jinja2 template extending `base.html`.

---

## Tests

```
tests/test_web_store.py        SummaryStore read-only methods
tests/test_web_routes.py       Flask test client — all pages, 404s, empty states
tests/test_web_file_preview.py whitelist, traversal rejection, size cap, JSON pretty-print
tests/fixtures/web_seed.py     seeds a minimal graph (1 source → 1 conv → 2 segs → 1 extract → 2 frags)
```

Run web tests only:

```bash
pytest tests/test_web_store.py tests/test_web_routes.py tests/test_web_file_preview.py -v
```
