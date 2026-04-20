# JARVIS — Architecture

## Overview

JARVIS processes raw data sources into a queryable semantic memory. The design is intentionally
local-first: all inference, embeddings, and storage run on your own machine.

```
Raw sources                     Ollama (inference + embedding)
  ChatGPT exports   ──ingest──▶  Chunks ──summarize──▶  SQLite (source of truth)
  Notion (planned)                                            │
  Slack (planned)                                        Qdrant (vector index)
                                                              │
                                                         retrieve / ask
```

---

## Storage Layers

Three layers, each with a distinct role:

| Layer | Technology | Role |
|---|---|---|
| **OUTPUTS/** | JSON + Markdown files | Raw artifacts from each run. Human-readable. |
| **SQLite** (`data/jarvis.db`) | SQLite | Source of truth for all structured summary records. Every field is stored here. |
| **Qdrant** | Qdrant (local Docker) | Vector retrieval index. Stores embeddings + minimal payload only. |

**Key principle:** Qdrant stores only the vector and a small payload (IDs, filter fields).
Full records are always fetched from SQLite by ID. Never treat Qdrant as the source of truth.

---

## Data Flow

### Ingestion

Raw source files are parsed, normalized to a canonical message schema, and split into
overlapping chunks (`user → assistant(s) → user`). Each chunk has a stable ID so re-ingesting
an updated export is safe — existing records are never duplicated.

### Chunk Summarization

Each chunk is summarized individually by a local LLM (Ollama). The last N chunk summaries
are passed as rolling context so the model understands conversational continuity without
re-summarizing earlier content.

Chunk summaries are the most granular retrieval unit — they score well for specific,
content-level queries.

### Segment Detection

Consecutive chunk summaries are grouped into topic segments by measuring cosine similarity
between adjacent summary embeddings. When similarity drops below a threshold, a new segment
begins. Each segment is then summarized with a single LLM call.

Segment summaries are higher-level — they score well for broad thematic queries. Both chunk
and segment summaries are indexed in Qdrant and retrieved together.

### Retrieval

A natural-language query is embedded and compared against all indexed vectors. Results are
ranked by cosine similarity, then full records are fetched from SQLite by ID.

---

## Embedding Strategy

Two distinct embedding approaches are used — do not conflate them:

| Use case | Text embedded | Why |
|---|---|---|
| **Retrieval indexing** | summary + bullets + action_items (multi-field) | Richer signal for semantic search |
| **Segment boundary detection** | summary text only | Multi-field text flattens inter-chunk similarity, making boundary detection unreliable |

Segment boundary detection always embeds on the fly from raw summary text. It never reuses
the Qdrant retrieval vectors.

---

## Local Services

| Service | Purpose | How to start |
|---|---|---|
| **Ollama** | LLM inference + embeddings | `ollama serve` |
| **Qdrant** | Vector search | `docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant` |

Ollama is required for every pipeline run. Qdrant is only required for `--persist` and `retrieve`.

---

## Models in Use

| Role | Model | Notes |
|---|---|---|
| Inference (summarization) | `gemma4:31b` | Set via `LOCAL_MODEL_NAME` in `.env` |
| Embeddings | `qwen3-embedding` | Set via `EMBEDDING_MODEL` in `.env` |

---

## SQLite Schema

The database lives at `data/jarvis.db` (configurable via `JARVIS_DB_PATH`). Current schema
version: **3**. Migrations are applied automatically on startup.

Key columns in the `summaries` table:

| Column | Description |
|---|---|
| `source_kind` | `conversation` \| `ai_chat_chunk` \| `ai_chat_segment` |
| `chunk_id` | Set for chunk summaries |
| `chunk_index` | Ordering within a conversation |
| `parent_conversation_id` | Links chunks and segments back to their conversation |
| `segment_index` | Set for segment summaries |
| `segment_chunk_range` | e.g. `"c000-c018"` — which chunks this segment covers |
| `status` | `ok` \| `degraded` |
| `qdrant_point_id` | UUID of the corresponding Qdrant point (null if not persisted) |
