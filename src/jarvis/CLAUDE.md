# src/jarvis/CLAUDE.md — Backend Pipeline Context

Loaded automatically when working on backend code. Covers architecture, module responsibilities,
and non-obvious design decisions that must not be re-litigated without good reason.

---

## Module Map

| Module | Responsibility |
|---|---|
| `cli.py` | Entry point for all CLI commands. Wires config → clients → domain logic. |
| `config.py` | Reads ENV + config.yaml + profiles.yaml without environment mutation; returns the effective config dict. |
| `clients.py` | Shared SQLite, Qdrant, embedding, inference, and memory factories; accepts resolved configuration without reloading it. |
| `endpoints.py` | Shared validation for remote Qdrant URLs and legacy host/port endpoints. |
| `profiles.py` | Frozen profile context, read-only storage validation, temporary personal-only runtime guard. |
| `paths.py` | Shared managed-path confinement, portable identifiers, artifact preflight, and external-source provenance hashes. |
| `recovery.py` | CLI scoped invalidation, dependency cleanup, and durable per-profile retry intents. |
| `artifacts.py` | Atomic output writes, cache validation, and complete fragment-batch recovery. |
| `ollama.py` | `OllamaClient` — inference via Ollama HTTP API. |
| `embedder.py` | `EmbeddingClient` — embeddings via Ollama. `build_embedding_text()` for retrieval. |
| `summarizer.py` | Single-file conversation summarization. |
| `segment_summarizer.py` | `SegmentSummarizer` — per-segment summarization with rolling context. |
| `topic_detector.py` | `TopicDetector` — cosine-similarity boundary detection + topic summarization. |
| `store.py` | `SummaryStore` — SQLite persistence. Source of truth for all records. |
| `memory.py` | `MemoryLayer` — entity-shaped wrappers over `SummaryStore` + fragment-only Qdrant indexing. |
| `vector_store.py` | `VectorStore` — Qdrant client wrapper (search, upsert, delete). |
| `output_writer.py` | `OutputWriter` — writes `.json` + `.md` artifacts to disk. |
| `ingest/` | Ingestion adapters. Currently: ChatGPT export parser + segmenter. |

---

## Pipeline Order

Commands must be run in this sequence for a given conversation:

```
ingest chatgpt --persist             →  inbox/…/<conv_id>/  +  SQLite: source_files, conversations, segments
summarize-segments chatgpt --persist →  OUTPUTS/…/segment_summaries/  +  SQLite: segment_summaries
detect-topics chatgpt --persist      →  OUTPUTS/…/topic_summaries/  +  SQLite: topic_summaries, topic_segments
extract-segments chatgpt --persist   →  OUTPUTS/…/extracts/  +  SQLite: extracts, extract_statements
fragment-extracts chatgpt --persist --embed  →  OUTPUTS/…/fragments/  +  SQLite: fragments, links  +  Qdrant
```

`retrieve` and `answer` work across all indexed fragments regardless of source.

---

## Output Paths

Stable paths — no timestamps in production flows:

| Artifact | Path |
|---|---|
| Conversation summary (single-file) | `OUTPUTS/<source_basename>/` |
| Segment summaries | `OUTPUTS/<conv_id>/segment_summaries/<segment_id>.json\|.md` |
| Topic summaries | `OUTPUTS/<conv_id>/topic_summaries/topic_NNN.json` |
| Extracts | `OUTPUTS/<conv_id>/extracts/extract_NNN.json\|.md` |
| Fragments | `OUTPUTS/<conv_id>/fragments/segment_NNN/fragment_MMM.json\|.md` |
| SQLite DB | `data/jarvis.db` |

`JARVIS_OUTPUT_TIMESTAMP` / `JARVIS_OUTPUT_TS_FORMAT` are still wired but effectively unused
in the main pipeline flows. Do not reintroduce timestamps without explicit discussion.

---

## Persistence Layer

Email V1 adds `pst.py` (shared optional-parser adapter and text decoding), `ingest/email.py`
(read-only streaming and retry ledger), `email_store.py` (explicit migration and email tables),
and `email_cli.py` (profile-scoped operators). Email clients use `clients.create_email_store`.
Base schema v8 and its startup validation remain unchanged. The separately versioned email
extension is installed only by `email migrate --backup`, after creating/reopening/verifying a
SQLite backup, with transactional DDL and data-preservation checks. No automatic migrations.
`email init` explicitly initializes only a missing base database; ordinary email commands never
initialize it. Original bodies are immutable after successful import; future cleaning belongs
in a separate representation keyed by email ID. No received classification is inferred from a
delivery timestamp. No memory, Qdrant, inference, web, or cleanup-recovery operations occur.
Real migrations/imports remain a separate reviewed operator step; see `docs/email-ingestion.md`.

Email Processing V1 adds `email_cleaning.py` (pure deterministic rules), `email_processing.py`
(bounded per-email checkpoints), and `email_processing_store.py` (separate versioned results).
Use `clients.create_email_processing_store`. The explicit `email processing-migrate --backup`
reuses `email_store.migrate_extension`; its preservation fingerprint includes original emails.
Base v8/email v1 stay unchanged. Keep quoted and uncertain content in processed text; preserve all
section text with qualitative detection evidence. Current version successes skip on resume;
errors retry, failed force preserves prior success, and new algorithm versions retain old rows.
No PST, inference, vector, attachment, thread reconstruction or web work belongs in this stage.
See `docs/email-processing.md`; real mom processing requires review and explicit authorization.

Email Embedding Preparation V1 adds `email_preparation.py` (pure signature/footer recheck,
message/section chunks and lexical near decisions) and `email_preparation_store.py` (checkpoints,
exact shared contents, every source occurrence). Use `clients.create_email_preparation_store`.
`email preparation-migrate --backup` installs its independent v1 extension through the verified
backup helper, fingerprinting all original and processed rows. Bump preparation rules version
after behavior changes; sizes and processing version are part of the generation key. Exact
duplicates share text only; near matches remain separate. Each email commits atomically, errors
retry and failed regeneration retains prior chunks. Recheck upstream signature exclusions and
retain uncertain business facts. No original/processed writes, service clients, or web work.
See `docs/email-embedding-preparation.md`; do not run against real mom before operator review.

Two stores, distinct roles:

- **SQLite (`SummaryStore`)** — relational source of truth for all pipeline records. Schema v8: the ten entity tables plus `jobs` and `_jarvis_meta`. Always query SQLite for full records. Database paths come from the resolved profile.
- **Qdrant (`VectorStore`)** — fragment vector index only. Collection comes from the resolved profile: personal `jarvis_fragments`, mom `jarvis_mom_fragments`. Payload per point: `{fragment_id, parent_conversation_id, segment_id, conversation_date}`. Full records fetched from SQLite by `fragment_id` via `store.get_fragments_with_statements()`.

Phase 2 preserves `_SCHEMA_VERSION = 8` and the DDL. Existing databases validate read-only without
DDL, metadata writes, or journal-mode changes on the existing file. Only a genuinely new file is
initialized. Unsupported or incomplete schemas fail unchanged. Never wipe/rebuild databases or
recreate existing collections automatically. All `INSERT OR IGNORE` operations retain their IDs.

Use `jarvis.clients` for application client construction. Embeddings use
`embedding_ollama_base_url` (fallback `ollama_base_url`); inference uses `ollama_base_url`.
Qdrant supports optional `qdrant_url`/`qdrant_api_key`, with legacy host/port when URL is absent.
Never include raw Qdrant errors or API keys in logs, configuration display, or job errors.
Memory clients capture resolved configuration and create Qdrant/embedding clients lazily on first
use. SQLite-only persistence must not contact either service. Explicitly injected clients remain
supported. Phase 2 kept destructive cleanup behavior; Commit 2 replaces CLI cleanup below.
Phase 3 Commit 1 uses `ProfilePaths` to route CLI filesystem operations and ingestion from the
resolved context. Managed overrides remain inside the active inbox root. External sources are
read-only and cannot come from another profile's storage. Validate paths and artifact identifiers
before any mutation. CLI retrieval/answer use `read_only=True` client/store access; missing
databases fail without initialization. Commit 2 now enables all processing CLI commands with scoped cleanup/recovery;
the personal-only `serve` and Flask guards remain. `CleanupService` journals known vector IDs before
SQLite invalidation, confirms vector deletion, removes files, then clears the intent. Reruns without
force replay pending cleanup. Never discard a pending journal. Existing databases are invalidated
when necessary even without `--persist`; missing databases/collections are not created for cleanup.
New vector IDs are deterministic; recorded UUIDs remain unchanged. `qdrant_point_id` with null
`embedded_at` is pending indexing, not completion. Only confirmed upsert sets `embedded_at`.
Validated JSON repairs missing SQLite rows/statements/links/Markdown; fragment manifests retain
complete model batches. Schema v8 is unchanged. One writer per profile is required; web job
cleanup/recovery and profile routing remain deferred to Phase 4.
Direct web runner calls must enforce `require_personal_runtime` before creating a store or marking
a job running. CLI processing is profile-aware; that guard is now specific to the web boundary.

---

## Two Embedding Strategies — Do Not Conflate

This is the single most important non-obvious design decision in the codebase.

| Use case | Function | Text used | Why |
|---|---|---|---|
| **Fragment retrieval indexing** | `build_embedding_text()` in `embedder.py` | Fragment `text` field (title + cleaned statements) | Retrieval-clean; no raw code/prompt blobs |
| **Topic boundary detection** | `_build_embedding_text()` in `topic_detector.py` | Segment summary text only | Multi-field text flattens inter-segment similarity, producing a near-uniform distribution unsuitable for boundary detection |

Topic boundary detection always embeds on the fly from summary text. It never reuses Qdrant
vectors. This was a deliberate fix — using Qdrant vectors gave a mean similarity of ~0.46 across
all pairs. Summary-only gives ~0.70 mean with meaningful variance.

Qdrant is **fragment-only**. Segment summaries and topic summaries are never embedded or indexed in Qdrant.

---

## OllamaClient — Return Type Contract

Both methods return **3-tuples**. Always unpack before use — passing the tuple directly to the
next method is a silent type error.

```python
raw, is_degraded, warning = self._ollama.generate(prompt)
parsed, parse_degraded, parse_warning = self._ollama.parse_json_response(raw)
```

`generate(prompt) -> Tuple[str, bool, str]` — (raw response, degraded flag, warning message)
`parse_json_response(raw) -> Tuple[Dict, bool, str]` — (parsed dict, degraded flag, warning message)

---

## Output Document — `status` Field

The model never includes `status` in its JSON output. Always inject it explicitly:

```python
output_data = {
    **parsed,
    "status": "ok",   # default — model won't provide this
    ...
}
if is_degraded:
    output_data["status"] = "degraded"
```

Do not rely on the model to set `status`. Do not use `.get("status", "ok")` as a fallback —
explicitly set it before the degraded check.

---

## Resume Logic & `--force`

**Resume (default):** if a `.json` output file already exists on disk for a segment/topic,
skip the LLM call and load from disk. This makes interrupted runs safely resumable.

**`--force`:** wipes existing output files and SQLite records (and Qdrant point IDs for fragments) for the affected scope, then re-runs from scratch.

**Critical scoping rule:** `--force` with `--from-segment X --to-segment Y` only affects
segments in that range. Segments outside the range are never touched.

---

## Ingestion Layout (ChatGPT)

```
inbox/ai_chat/chatgpt/<conversation_id>/
  normalized.json          # canonical message list
  segment_manifest.json    # segment count, IDs, pending_tail
  segments/
    segment_000.json       # segment_id, segment_index, segment_text, message_ids
    segment_001.json
    ...
    pending_tail.json      # present only if last message has no reply yet — skip in summarization
```

Segmenting rule: `user → assistant(s) → user` overlap — segment N ends with the user message
that opens segment N+1. `pending_tail.json` must never be passed to the summarizer.

---

## Local Services

Service requirements depend on the operation:

- **Ollama** (`ollama serve`) — required when inference or embedding is performed
- **Qdrant** (`docker run -p 6333:6333 qdrant/qdrant`) — required for indexing, retrieval/answer, and explicit vector cleanup; SQLite-only persistence does not require it

Neither starts automatically.
