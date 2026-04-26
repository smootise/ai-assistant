# JARVIS Web App

## Overview

The web app is an operator console for running and monitoring the JARVIS pipeline from a browser.
It runs locally at `http://localhost:5000`.

V1 scope: browse data lineage · upload + ingest · launch extract-segments · launch fragment-extracts · track all pipeline jobs.
Not yet supported: retrieve/answer via the UI.

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
| `GET /jobs` | Jobs list | All jobs (ingest + extract + fragment), newest first |
| `GET /jobs/<id>` | Job status | Status, timestamps, job-type-specific result links or error traceback |
| `GET /conversations/<id>/extract` | Extract form | Launch extract-segments for a conversation |
| `POST /conversations/<id>/extract` | Extract submit | Validates, creates job, starts daemon thread → 303 to `/jobs/<id>` |
| `GET /conversations/<id>/fragment` | Fragment form | Launch fragment-extracts (blocked if no extracts exist) |
| `POST /conversations/<id>/fragment` | Fragment submit | Validates (embed→persist, range, prereqs), creates job → 303 to `/jobs/<id>` |

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

## Pipeline job launch flow

The extract and fragment jobs follow the same pattern as the ingest job:

1. **Form GET** renders the form page. Fragment form page is blocked (warning only) when no
   extracts exist for the conversation.
2. **Form POST** validates inputs server-side and creates a `jobs` row (`job_type` =
   `extract_segments` or `fragment_extracts`), then spawns a daemon thread and redirects to
   `/jobs/<id>`.
3. **Daemon thread** (`extract_runner.run_extract_job` or `fragment_runner.run_fragment_job`)
   calls the core pipeline classes directly — no subprocess. Opens its own `SummaryStore`
   connection, marks job running → succeeded/failed.
4. **Job detail page** auto-refreshes while pending/running. On success, shows job options and
   links to the conversation and to each created extract/fragment (capped at 200 IDs in the
   stored result blob).

### Validation rules (extract + fragment forms)

| Rule | Response on failure |
|---|---|
| conversation must exist | 404 |
| `from_segment` / `to_segment` must be integers ≥ 0 | 400 + flash |
| `from_segment` ≤ `to_segment` when both given | 400 + flash |
| `embed` requires `persist` | 400 + flash |
| fragment: no extracts exist for conversation | 400 + flash (POST); warning page (GET) |

### job_type values

| Value | Meaning |
|---|---|
| `ingest_chatgpt` | Upload + ingest flow |
| `extract_segments` | Extract attributed statements per segment |
| `fragment_extracts` | Fragment extracts into retrieval units; optionally embed in Qdrant |

`input_metadata` for `extract_segments`: `{conversation_id, from_segment, to_segment, force, persist}`.
`input_metadata` for `fragment_extracts`: `{conversation_id, from_segment, to_segment, force, persist, embed}`.

---

## Code layout

```
src/jarvis/web/
  __init__.py          exports create_app()
  app.py               Flask factory — registers blueprints, Jinja filters, store factory
  services.py          thin service layer — one function per page, returns plain dicts
  file_preview.py      ID-first whitelisted file reader
  ingest_runner.py     module-level run_ingest_job() called from daemon threads
  extract_runner.py    module-level run_extract_job() called from daemon threads
  fragment_runner.py   module-level run_fragment_job() called from daemon threads
  routes/
    dashboard.py       GET /
    sources.py         GET /sources, /sources/<id>, /sources/<id>/raw
    conversations.py   GET /conversations, /conversations/<id>
    segments.py        GET /segments/<id>, /segments/<id>/raw
    extracts.py        GET /extracts/<id>, /extracts/<id>/raw
    fragments.py       GET /fragments/<id>, /fragments/<id>/raw
    uploads.py         GET /upload, POST /upload
    jobs.py            GET /jobs, /jobs/<id>
    pipeline_jobs.py   GET+POST /conversations/<id>/extract, /conversations/<id>/fragment
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
    pipeline_extract_form.html
    pipeline_fragment_form.html
    jobs_list.html / job_detail.html
    404.html
  static/
    styles.css         single stylesheet — system fonts, no framework
```

---

## Architecture

The web layer orchestrates pipeline jobs in-process via runner modules:

```
Browser
  └── Flask routes (routes/)
        ├── services.py  (read orchestration; normalizes missing data to None/[])
        │     └── SummaryStore public methods → SQLite  data/jarvis.db
        └── *_runner.py  (job execution in daemon threads)
              ├── SegmentExtractor / Fragmenter / MemoryLayer  (pipeline classes)
              ├── SummaryStore  (own connection per thread)
              └── VectorStore (Qdrant, only when embed=True)
```

The web layer never calls `SummaryStore._connect()` directly — all queries go through public
methods. New read queries belong in `store.py` as public `SummaryStore` methods, not in routes
or services. Pipeline classes are invoked only from `*_runner.py`, never directly from routes.

---

## Adding another pipeline stage as a job

To expose a new CLI pipeline stage (e.g. `summarize-segments`) as a web job:

1. **Store** — add any needed read queries as public `SummaryStore` methods in `store.py`.
2. **Runner** — create `src/jarvis/web/<stage>_runner.py` with a module-level
   `run_<stage>_job(job_id, conversation_id, options, config)`. Open a fresh `SummaryStore`,
   call `mark_job_running`, run the pipeline class, call `mark_job_succeeded(result)` or
   `mark_job_failed(traceback)`.
3. **Route** — add GET + POST handlers in `routes/pipeline_jobs.py` (or a new blueprint).
   Validate inputs, create the job row (`store.create_job("<job_type>", input_metadata)`),
   spawn a daemon thread running the runner, redirect to `/jobs/<id>`.
4. **Template** — add a form template extending `base.html`.
5. **job_detail.html** — add a branch for the new `job_type` in the result section.
6. **Tests** — add route tests (form GET, valid POST, validation errors, 404 on bad ID) and
   a job_detail rendering test for succeeded + failed states.
7. **Docs** — update this file: routes table, job_type values, `input_metadata` shape.

Planned additions:
- **Retrieve/answer**: query form + citation drill-down using existing `retrieve` and `answer` CLI logic.

---

## Tests

```
tests/test_web_store.py           SummaryStore read-only methods
tests/test_web_routes.py          Flask test client — all pages, 404s, empty states
tests/test_web_file_preview.py    whitelist, traversal rejection, size cap, JSON pretty-print
tests/test_web_uploads.py         upload form, validation, job creation, thread spawn
tests/test_web_jobs.py            job detail page — all statuses, auto-refresh logic
tests/test_web_pipeline_jobs.py   extract + fragment forms, validation, job creation, job_detail rendering
tests/fixtures/web_seed.py        seeds a minimal graph (1 source → 1 conv → 2 segs → 1 extract → 2 frags)
```

Run web tests only:

```bash
pytest tests/test_web_store.py tests/test_web_routes.py tests/test_web_file_preview.py \
       tests/test_web_uploads.py tests/test_web_jobs.py tests/test_web_pipeline_jobs.py -v
```
