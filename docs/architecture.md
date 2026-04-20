# JARVIS — Architecture

## Overview

JARVIS processes raw data sources into a queryable semantic memory. The design is intentionally
local-first: all inference, embeddings, and storage run on your own machine.

```
Raw sources                       Ollama (inference + embedding)
  ChatGPT exports   ──ingest──▶  Segments ──summarize──▶  SQLite (source of truth)
  Notion (planned)                                              │
  Slack (planned)                                          Qdrant (vector index)
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

**Key principle:** Qdrant stores only the vector and a small payload (IDs, filter fields).
Full records are always fetched from SQLite by ID. Never treat Qdrant as the source of truth.

---

## Data Flow

### Ingestion

Raw source files are parsed, normalized to a canonical message schema, and split into
overlapping segments (`user → assistant(s) → user`). Each segment has a stable ID so
re-ingesting an updated export is safe — existing records are never duplicated.

### Segment Summarization

Each segment is summarized individually by a local LLM (Ollama). The last N segment summaries
are passed as rolling context so the model understands conversational continuity without
re-summarizing earlier content.

Segment summaries are the most granular retrieval unit — they score well for specific,
content-level queries.

### Topic Detection

Consecutive segment summaries are grouped into topics by measuring cosine similarity
between adjacent summary embeddings. When similarity drops below a threshold, a new topic
begins. Each topic is then summarized with a single LLM call.

Topic summaries are higher-level — they score well for broad thematic queries. Both segment
and topic summaries are indexed in Qdrant and retrieved together.

### Retrieval

A natural-language query is embedded and compared against all indexed vectors. Results are
ranked by cosine similarity, then full records are fetched from SQLite by ID.

---

## Embedding Strategy

Two distinct embedding approaches are used — do not conflate them:

| Use case | Text embedded | Why |
|---|---|---|
| **Retrieval indexing** | summary + bullets + action_items (multi-field) | Richer signal for semantic search |
| **Topic boundary detection** | summary text only | Multi-field text flattens inter-segment similarity, making boundary detection unreliable |

Topic boundary detection always embeds on the fly from raw summary text. It never reuses
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
version: **4**. Schema is applied fresh on startup — no migrations from prior versions.

Key columns in the `summaries` table:

| Column | Description |
|---|---|
| `source_kind` | `conversation` \| `ai_chat_segment` \| `ai_chat_topic` |
| `segment_id` | Set for segment summaries (e.g. `686c5e2b_s003`) |
| `segment_index` | Ordering within a conversation (segment summaries) |
| `parent_conversation_id` | Links segments and topics back to their conversation |
| `topic_index` | Set for topic summaries |
| `topic_segment_range` | e.g. `"s000-s018"` — which segments this topic covers |
| `status` | `ok` \| `degraded` |
| `qdrant_point_id` | UUID of the corresponding Qdrant point (null if not persisted) |
