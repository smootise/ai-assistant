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
| `segment_summarizer.py` | `SegmentSummarizer` — per-segment summarization with rolling context. |
| `topic_detector.py` | `TopicDetector` — cosine-similarity boundary detection + topic summarization. |
| `store.py` | `SummaryStore` — SQLite persistence. Source of truth for all records. |
| `memory.py` | `MemoryPersister` — writes to SQLite + indexes in Qdrant. |
| `vector_store.py` | `VectorStore` — Qdrant client wrapper (search, upsert, delete). |
| `output_writer.py` | `OutputWriter` — writes `.json` + `.md` artifacts to disk. |
| `ingest/` | Ingestion adapters. Currently: ChatGPT export parser + segmenter. |

---

## Pipeline Order

Commands must be run in this sequence for a given conversation:

```
ingest chatgpt            →  normalize + segment  →  inbox/ai_chat/chatgpt/<conv_id>/
summarize-segments chatgpt →  segment summaries   →  OUTPUTS/<conv_id>/segment_summaries/
detect-topics chatgpt      →  topic summaries     →  OUTPUTS/<conv_id>/topic_summaries/
```

`retrieve` and `answer` work across all persisted data regardless of source.

---

## Output Paths

Stable paths — no timestamps in production flows:

| Artifact | Path |
|---|---|
| Conversation summary | `OUTPUTS/<source_basename>/` |
| Segment summaries | `OUTPUTS/<conv_id>/segment_summaries/<segment_id>.json\|.md` |
| Topic summaries | `OUTPUTS/<conv_id>/topic_summaries/topic_NNN.json` |
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
  `segment_id`, `parent_conversation_id`). Full records are always fetched from SQLite by ID.

SQLite schema is at **version 4**. Schema was rebuilt cleanly at v4 — no backward-compat
migrations needed (databases were emptied). New columns must be added via a `_MIGRATION_V5`
block appended to `_MIGRATIONS` in `store.py`.

---

## Two Embedding Strategies — Do Not Conflate

This is the single most important non-obvious design decision in the codebase.

| Use case | Function | Text used | Why |
|---|---|---|---|
| **Retrieval indexing** | `build_embedding_text()` in `embedder.py` | Multi-field: summary + bullets + action_items | Richer signal for semantic search |
| **Topic boundary detection** | `_build_embedding_text()` in `topic_detector.py` | Summary text only | Multi-field text flattens inter-segment similarity, producing a near-uniform distribution unsuitable for boundary detection |

Topic boundary detection always embeds on the fly from summary text. It never reuses Qdrant
vectors. This was a deliberate fix — using Qdrant vectors gave a mean similarity of ~0.46 across
all pairs. Summary-only gives ~0.70 mean with meaningful variance.

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

**`--force`:** wipes existing output files and SQLite/Qdrant records for the affected scope,
then re-runs from scratch.

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

Both must be running before any backend command works:

- **Ollama** (`ollama serve`) — required for inference and embedding on every run
- **Qdrant** (`docker run -p 6333:6333 qdrant/qdrant`) — required only for `--persist` and `retrieve`

Neither starts automatically.
