# JARVIS — Architecture

## Overview

JARVIS processes raw data sources into a queryable semantic memory. The design is intentionally
local-first: all inference, embeddings, and storage run on your own machine.

```
Raw sources                       Ollama (inference + embedding)
  ChatGPT exports   ──ingest──▶  Segments ──summarize──▶  SQLite (relational, v7)
  Notion (planned)     ──extract-segments──▶  Statements       │
  Slack (planned)         ──fragment-extracts──▶  Fragments ──▶ Qdrant (fragment vectors)
                                                                │
                                                           retrieve / answer
```

---

## Storage Layers

Three layers, each with a distinct role:

| Layer | Technology | Role |
|---|---|---|
| **OUTPUTS/** | JSON + Markdown files | Raw artifacts from each run. Human-readable. |
| **SQLite** (`data/jarvis.db`) | SQLite | Source of truth for all structured summary records. Every field is stored here. |
| **Qdrant** | Qdrant (local Docker) | Vector retrieval index. Stores embeddings + minimal payload only. |

**Key principle:** Qdrant stores only the vector and a small payload (`fragment_id`, `parent_conversation_id`, `segment_id`, `conversation_date`). Full records are always fetched from SQLite by ID via `get_fragments_with_statements`. Never treat Qdrant as the source of truth.

---

## Data Flow

### Ingestion

Raw source files are parsed, normalized to a canonical message schema, and split into
overlapping segments (`user → assistant(s) → user`). Each segment has a stable ID so
re-ingesting an updated export is safe — existing records are never duplicated.

### Segment Summarization

Each segment is summarized individually by a local LLM (Ollama). The last N segment summaries
are passed as rolling context so the model understands conversational continuity without
re-summarizing earlier content. Summaries are stored in `segment_summaries` (SQLite only — not indexed in Qdrant).

### Topic Detection

Consecutive segment summaries are grouped into topics by measuring cosine similarity
between adjacent summary embeddings. When similarity drops below a threshold, a new topic
begins. Each topic is then summarized with a single LLM call. Topic summaries are stored in
`topic_summaries` + `topic_segments` (SQLite only — not indexed in Qdrant).

### Extraction

Each segment's text is processed by the extractor, which identifies all informational content
as attributed statements (`speaker` + `text`). Statements are stored in `extracts` +
`extract_statements` (SQLite only).

### Fragmentation + Embedding

Extracted statements are grouped into topically coherent fragments — the actual retrieval unit.
Each fragment's retrieval text is embedded and upserted to Qdrant. The `qdrant_point_id` is
written back to the `fragments` table so the SQLite ↔ Qdrant link is always recoverable.

### Retrieval

A natural-language query is embedded and compared against all fragment vectors in Qdrant.
Results are ranked by cosine similarity, then full records (fragment + statements + segment
context) are fetched from SQLite by ID via `get_fragments_with_statements`.

---

## Embedding Strategy

Two distinct embedding approaches are used — do not conflate them:

| Use case | Text embedded | Why |
|---|---|---|
| **Fragment retrieval indexing** | fragment `text` field (title + cleaned statements) | Retrieval-clean; no raw code/prompt blobs |
| **Topic boundary detection** | segment summary text only | Multi-field text flattens inter-segment similarity, making boundary detection unreliable |

Topic boundary detection always embeds on the fly from raw summary text. It never reuses
the Qdrant retrieval vectors. Qdrant is fragment-only — segment and topic summaries are never embedded.

---

## Local Services

| Service | Purpose | How to start |
|---|---|---|
| **Ollama** | LLM inference + embeddings | `ollama serve` |
| **Qdrant** | Vector search | `docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant` |

Ollama is required for every pipeline run. Qdrant is only required for `fragment-extracts --embed` and `retrieve`/`answer`.

---

## Models in Use

| Role | Model | Notes |
|---|---|---|
| Inference (summarization) | `gemma4:31b` | Set via `LOCAL_MODEL_NAME` in `.env` |
| Embeddings | `qwen3-embedding` | Set via `EMBEDDING_MODEL` in `.env` |

---

## SQLite Schema

The database lives at `data/jarvis.db` (configurable via `JARVIS_DB_PATH`). Current schema
version: **7**. Schema is applied fresh on startup — no migrations from prior versions (wipe
and rebuild).

Table hierarchy (each table has a deterministic `*_id` primary key):

```
source_files                       SHA-256-keyed; raw export + normalized.json metadata
  └── conversations                one per ChatGPT conversation
        └── segments               one per segment (stores full segment_text for citations)
              ├── segment_summaries   LLM-generated per-segment summary
              ├── extracts            one per segment (LLM extraction run)
              │     ├── extract_statements   one row per attributed statement
              │     └── fragments            one per topical fragment
              │           └── fragment_statement_links  →  extract_statements
              └── topic_summaries     LLM-generated per-topic summary
                    └── topic_segments  →  segments  (many-to-many)
```

ID determinism: `segment_id = f"{conv_id}_s{idx:03d}"`, `extract_id = f"{segment_id}_x"`,
`statement_id = f"{extract_id}_st{idx:04d}"`, `fragment_id = f"{extract_id}_f{idx:03d}"`.
`INSERT OR IGNORE` on stable IDs makes every `--persist` run idempotent.

**Qdrant payload** per fragment point: `{fragment_id, parent_conversation_id, segment_id, conversation_date}`. All other data is in SQLite.

---

## Web Layer (V1 — Read-Only)

A Flask + Jinja2 operator console (`src/jarvis/web/`) sits above `SummaryStore` and provides
browser-based browsing of the full entity graph. It is strictly read-only in V1 — no pipeline
invocations, no SQLite writes, no Qdrant queries.

```
Browser
  └── Flask routes (src/jarvis/web/routes/)
        └── services.py  (thin orchestration, missing-data normalization)
              └── SummaryStore (public methods only — never _connect() directly)
                    └── SQLite (data/jarvis.db)
```

**File preview** is ID-first: paths are resolved from SQLite entity metadata, never from
user-supplied strings. Every resolved path is re-validated against an approved whitelist
(`OUTPUTS/`, `inbox/ai_chat/`) before reading. Symlink escapes and path traversal are rejected.
Size cap: 2 MB. Allowed extensions: `.json`, `.md`, `.txt`.

**Start the web UI:**

```bash
python -m jarvis.cli serve          # default: 127.0.0.1:5000
python -m jarvis.cli serve --port 8080 --debug
```
