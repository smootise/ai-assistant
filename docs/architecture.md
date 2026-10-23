# JARVIS — Architecture

## Overview

JARVIS processes raw data sources into a queryable semantic memory. The design is intentionally
local-first: all inference, embeddings, and storage run on your own machine.

```
Raw sources                       Ollama (inference + embedding)
  ChatGPT exports   ──ingest──▶  Segments ──summarize──▶  SQLite (relational, v8)
  Notion (planned)     ──extract-segments──▶  Statements       │
  Slack (planned)         ──fragment-extracts──▶  Fragments ──▶ Qdrant (fragment vectors)
                                                                │
                                                           retrieve / answer
```

---

## Storage Layers

Phase 3 adds filesystem routing, scoped cleanup, interrupted-run recovery, and multi-profile CLI
processing to the Phase 2 storage clients. Personal retains its existing resources; web serving
remains personal-only until Phase 4. See [Profiles](profiles.md) for the registry, validation rules, and mandatory
backup/compatibility checkpoint before real-data deployment. No SQLite schema change is included.

Three layers, each with a distinct role:

| Layer | Technology | Role |
|---|---|---|
| **OUTPUTS/** | JSON + Markdown files | Raw artifacts from each run. Human-readable. |
| **SQLite** (`data/jarvis.db`) | SQLite | Source of truth for all structured summary records. Every field is stored here. |
| **Qdrant** | Qdrant (local Docker) | Vector retrieval index. Stores embeddings + minimal payload only. |

**Key principle:** Qdrant stores only the vector and a small payload (`fragment_id`, `parent_conversation_id`, `segment_id`, `conversation_date`). Full records are always fetched from SQLite by ID via `get_fragments_with_statements`. Never treat Qdrant as the source of truth.

---

All application client construction goes through `jarvis.clients` using the resolved configuration.
`jarvis.paths` supplies managed-path boundaries and source validation shared by CLI routing and
ingestion. Commands validate existing artifact trees and identifiers before file mutations; roots
come from the active `ProfileContext`. External sources remain read-only and retain path/hash
provenance. Identical conversation IDs occupy independent profile directories.
Memory clients open SQLite immediately and construct Qdrant and embedding clients only on first
use, then reuse them. SQLite-only persistence in CLI commands and web jobs does not require Qdrant
or an embedding service. Topic detection embeds summary text directly without a Qdrant probe.
Each SQLite client uses its profile's database path; each Qdrant client uses its profile's collection.
Personal remains `jarvis_fragments`; mom is `jarvis_mom_fragments`. Collections are created lazily
only when absent and are never automatically reset. Payloads are unchanged. Newly indexed fragments
use deterministic UUIDs; existing indexed fragments retain their recorded UUIDs.

## Cleanup and recovery

`jarvis.recovery.CleanupService` owns CLI invalidation. It selects actual input segment IDs in an
inclusive range, validates the complete deletion set, and writes a durable intent under the active
output root at `.jarvis-recovery/cleanup_<digest>.json` before mutating stores or files. The intent
binds to the profile, database, output root, collection, and a hash of the Qdrant endpoint; it contains
no service credentials. Cleanup never initializes an absent SQLite database or Qdrant collection.

The dependency graph is segment summary -> all conversation topics, and extract -> statements ->
fragments -> statement links/vectors/artifacts. Fragment invalidation preserves extracts/statements;
summary invalidation preserves independent extracts/fragments. `SummaryStore.invalidate_stage`
removes relational dependents in one scoped SQLite transaction. Required invalidation applies even
without `--persist`. Standalone summaries have no relational persistence and clean only their pair
of artifacts.

After the journal is durable, cleanup removes SQLite records, confirms Qdrant deletion with
`wait=True`, removes artifacts, then deletes the journal. If any step fails, the journal retains
the known vector IDs and is replayed idempotently by the same command without `--force`. A narrower
retry range is rejected if it excludes pending work. Other processing commands and search refuse
to proceed while cleanup is pending; do not delete or edit a recovery journal to bypass this guard.
This deliberately favors temporary missing data over stale retrieval. There is no rollback.

Indexing records its intended point ID in `fragments.qdrant_point_id` with `embedded_at=NULL`
before upsert. New IDs are UUIDv5 of `jarvis:fragment:<fragment_id>` in the URL namespace; a recorded
legacy UUID is reused. A successful upsert followed by an interrupted SQLite completion update can
therefore be retried at the same ID. Only a confirmed upsert sets `embedded_at`; search excludes
unfinished indexing. A subsequent cleanup can also find and remove these pending point IDs.
No SQLite schema, existing IDs, payloads, models, or vectors are migrated.

`jarvis.artifacts` supplies flushed, fsynced temporary-file writes followed by atomic replacement,
and validates cached outputs before reuse. Successful JSON can repair missing SQLite rows,
statements, links, and Markdown. A fragment manifest publishes the complete model result before
individual files, allowing interrupted rendering to finish without repeating inference. Topic input
hashes invalidate changed boundary inputs/thresholds; unchanged topics resume individually.
Oversized extracts checkpoint successful model batches as `.batch_<index>.json`, validated against
an input/model/schema fingerprint. Automatic fragment recovery retains these checkpoints; explicit
force and upstream extraction invalidation clear them. Failed batches make the segment partial.
If validated JSON disagrees with partial/stale persisted content, cleanup preserves that JSON while
invalidating the affected records and dependencies, then repairs persistence without model calls.
Cleanup selects generated artifact names and refuses to delete an original export recorded in
normalized provenance or the conversation's SQLite source metadata.

Recovery assumes one writer per profile and stable filesystem/service boundaries during an
operation; concurrent CLI/web writers are unsupported. Journals and fragment manifests are required
recovery data and belong in backups. Lost/corrupt journals, manual database replacement/removal,
and storage/device failure require operator recovery; fsync/rename is not a cross-store transaction
or a guarantee against hardware loss. Historical orphan vectors created by older random-ID failures
cannot be inferred automatically and are not migrated or swept. Web job cleanup/recovery remains
outside this CLI rollout.

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

Inference uses `OLLAMA_BASE_URL`. Embeddings use `EMBEDDING_OLLAMA_BASE_URL` when provided,
otherwise the inference endpoint. This supports desktop inference and a separate TrueNAS embedding
service without changing the embedding model or existing vectors. Qdrant can use an HTTP(S)
`QDRANT_URL` plus optional `QDRANT_API_KEY`, or the existing host/port configuration. See
[Profiles](profiles.md#phase-2-storage-and-service-construction) for validation and precedence.

---

## Models in Use

| Role | Model | Notes |
|---|---|---|
| Inference (summarization) | `gemma4:31b` | Set via `LOCAL_MODEL_NAME` in `.env` |
| Embeddings | `qwen3-embedding` | Set via `EMBEDDING_MODEL` in `.env` |

---

## SQLite Schema

The database lives at `data/jarvis.db` (configurable via `JARVIS_DB_PATH`). Current schema
version: **8**. Only a genuinely missing database file is initialized. Existing databases are
validated through a read-only connection without changing their schema, metadata, or journal mode.
Required structures are checked against an in-memory v8 reference; harmless additional objects
are tolerated. Incompatible or incomplete files fail unchanged. There are no migrations or
automatic rebuilds.
CLI retrieval and answer use the explicit `read_only=True` store/factory mode. Missing databases
fail without filesystem initialization; existing files still undergo the same v8 compatibility
validation. Subsequent connections use SQLite read-only/query-only modes. Search never creates
missing Qdrant collections.

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

jobs                               upload + ingest job tracking (independent of the entity graph)
```

ID determinism: `segment_id = f"{conv_id}_s{idx:03d}"`, `extract_id = f"{segment_id}_x"`,
`statement_id = f"{extract_id}_st{idx:04d}"`, `fragment_id = f"{extract_id}_f{idx:03d}"`.
`INSERT OR IGNORE` on stable IDs makes every `--persist` run idempotent.

**Qdrant payload** per fragment point: `{fragment_id, parent_conversation_id, segment_id, conversation_date}`. All other data is in SQLite.

---

## Jobs Layer

The `jobs` table tracks all pipeline operations triggered by the web UI. Each job has:
`status ∈ {pending, running, succeeded, failed}`, ISO-8601 timestamps (`created_at`,
`started_at`, `finished_at`), a `job_type` string, the input metadata (JSON), and either
a result dict or an error traceback.

Current `job_type` values: `ingest_chatgpt`, `extract_segments`, `fragment_extracts`.

Each job runs in a **daemon thread** inside the Flask process via a dedicated runner module
(`ingest_runner.py`, `extract_runner.py`, `fragment_runner.py`). Each runner opens a fresh
`SummaryStore` connection (SQLite WAL allows concurrent readers and one writer) and calls
pipeline classes directly — no subprocess. Jobs interrupted by a server shutdown remain
stuck in `running` — a startup sweep is deferred.

The `jobs` table is independent of the entity graph and is never referenced by foreign keys
from other tables.

---

## Web Layer

A Flask + Jinja2 operator console (`src/jarvis/web/`) for running and monitoring the pipeline
from a browser. Sits above `SummaryStore` public methods — never touches `_connect()`.

V1 features: browse the source → conversation → segment → extract → fragment lineage;
upload ChatGPT export files; launch extract-segments and fragment-extracts for any conversation;
search fragments by semantic query; generate RAG answers with clickable citations;
track all pipeline jobs; inspect raw JSON/text artifacts. The web layer reads from Qdrant
only for the `/search` and `/answer` routes via the same `run_retrieval`/`generate_answer`
helpers used by the CLI.

See [docs/webapp.md](webapp.md) for the full route map, job launch flow, validation rules,
code layout, and extension guide.
