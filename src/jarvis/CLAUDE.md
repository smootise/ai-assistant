# src/jarvis/CLAUDE.md — Backend Pipeline Context

Loaded automatically when working on backend code. Covers architecture, module responsibilities,
and non-obvious design decisions that must not be re-litigated without good reason.

---

## Module Map

| Module | Responsibility |
|---|---|
| `cli.py` | Entry point for all CLI commands. Wires config → clients → domain logic. |
| `config.py` | Loads ENV + config.yaml, returns a single config dict. |
| `ollama.py` | `OllamaClient` — inference via Ollama HTTP API. |
| `embedder.py` | `EmbeddingClient` — embeddings via Ollama. `build_embedding_text()` for retrieval. |
| `summarizer.py` | Single-file conversation summarization. |
| `chunk_summarizer.py` | `ChunkSummarizer` — per-chunk summarization with rolling context. |
| `segment_detector.py` | `SegmentDetector` — cosine-similarity boundary detection + segment summarization. |
| `store.py` | `SummaryStore` — SQLite persistence. Source of truth for all records. |
| `memory.py` | `MemoryPersister` — writes to SQLite + indexes in Qdrant. |
| `vector_store.py` | `VectorStore` — Qdrant client wrapper (search, upsert, delete). |
| `output_writer.py` | `OutputWriter` — writes `.json` + `.md` artifacts to disk. |
| `ingest/` | Ingestion adapters. Currently: ChatGPT export parser + chunker. |

---

## Pipeline Order

Commands must be run in this sequence for a given conversation:

```
ingest chatgpt          →  normalize + chunk  →  inbox/ai_chat/chatgpt/<conv_id>/
summarize-chunks chatgpt →  chunk summaries   →  OUTPUTS/<conv_id>/chunk_summaries/
detect-segments chatgpt  →  segment summaries →  OUTPUTS/<conv_id>/segment_summaries/
```

`retrieve` and (future) RAG answer generation work across all persisted data regardless of source.

---

## Output Paths

Stable paths — no timestamps in production flows:

| Artifact | Path |
|---|---|
| Conversation summary | `OUTPUTS/<source_basename>/` |
| Chunk summaries | `OUTPUTS/<conv_id>/chunk_summaries/<chunk_id>.json\|.md` |
| Segment summaries | `OUTPUTS/<conv_id>/segment_summaries/segment_NNN.json\|.md` |
| SQLite DB | `data/jarvis.db` |

`JARVIS_OUTPUT_TIMESTAMP` / `JARVIS_OUTPUT_TS_FORMAT` are still wired but effectively unused
in the main pipeline flows. Do not reintroduce timestamps without explicit discussion.

---

## Persistence Layer

Two stores, distinct roles:

- **SQLite (`SummaryStore`)** — canonical record for every summary. Every field from the output
  document is stored here. Always query SQLite for full records.
- **Qdrant (`VectorStore`)** — vector index only. Stores embedding + minimal payload
  (`summary_id`, `source_file`, `source_kind`, `status`, `created_at`, `model`,
  `chunk_id`, `parent_conversation_id`). Full records are always fetched from SQLite by ID.

SQLite schema is at **version 3**. Migrations live in `_MIGRATION_V2` and `_MIGRATION_V3`
in `store.py`. Add new columns via a new `_MIGRATION_V4` block — never alter `_DDL` directly
for columns that don't exist on old databases (indexes on new columns must also go in the
migration, not in `_DDL`).

---

## Two Embedding Strategies — Do Not Conflate

This is the single most important non-obvious design decision in the codebase.

| Use case | Function | Text used | Why |
|---|---|---|---|
| **Retrieval indexing** | `build_embedding_text()` in `embedder.py` | Multi-field: summary + bullets + action_items | Richer signal for semantic search |
| **Segment boundary detection** | `_build_embedding_text()` in `segment_detector.py` | Summary text only | Multi-field text flattens inter-chunk similarity, producing a near-uniform distribution unsuitable for boundary detection |

Segment boundary detection always embeds on the fly from summary text. It never reuses Qdrant
vectors. This was a deliberate fix — using Qdrant vectors gave a mean similarity of ~0.46 across
all pairs (100 segments out of 104 chunks). Summary-only gives ~0.70 mean with meaningful variance.

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

**Resume (default):** if a `.json` output file already exists on disk for a chunk/segment,
skip the LLM call and load from disk. This makes interrupted runs safely resumable.

**`--force`:** wipes existing output files and SQLite/Qdrant records for the affected scope,
then re-runs from scratch.

**Critical scoping rule:** `--force` with `--from-chunk X --to-chunk Y` only affects chunks
in that range. Chunks outside the range are never touched. Do not implement `--force` in a way
that wipes the full conversation when a range is specified.

---

## Ingestion Layout (ChatGPT)

```
inbox/ai_chat/chatgpt/<conversation_id>/
  normalized.json        # canonical message list
  chunk_manifest.json    # chunk count, IDs, pending_tail
  chunks/
    chunk_000.json       # chunk_id, chunk_index, chunk_text, message_ids
    chunk_001.json
    ...
    pending_tail.json    # present only if last message has no reply yet — skip in summarization
```

Chunking rule: `user → assistant(s) → user` overlap — chunk N ends with the user message
that opens chunk N+1. `pending_tail.json` must never be passed to the summarizer.

---

## Local Services

Both must be running before any backend command works:

- **Ollama** (`ollama serve`) — required for inference and embedding on every run
- **Qdrant** (`docker run -p 6333:6333 qdrant/qdrant`) — required only for `--persist` and `retrieve`

Neither starts automatically.
