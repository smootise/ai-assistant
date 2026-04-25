# JARVIS — Local-first AI Assistant for PM Workflows

![CI](https://github.com/smootise/ai-assistant/actions/workflows/ci.yml/badge.svg)

## What is JARVIS?

As a Product Manager, you accumulate a constant stream of information — Notion specs, Slack threads, meeting transcripts, ChatGPT conversations, decision logs. Most of it disappears into the noise. JARVIS is a local, privacy-first assistant that ingests all of it, indexes it semantically, and lets you ask questions in plain language:

- *"Why did we decide to use Qdrant over Pinecone?"*
- *"What action items were assigned to me in the last sprint planning?"*
- *"What was the reasoning behind the segmentation strategy?"*

Everything runs on your own machine. No data leaves your environment.

---

## How it works

```
Raw data (ChatGPT export, Notion, Slack, ...)
    ↓  ingest --persist
Normalized segments → SQLite (source_files, conversations, segments)
    ↓  summarize-segments --persist
Segment summaries → SQLite (segment_summaries)
    ↓  detect-topics --persist
Topic summaries → SQLite (topic_summaries)
    ↓  extract-segments --persist
Attributed statements → SQLite (extracts, extract_statements)
    ↓  fragment-extracts --persist --embed
Fragments → SQLite (fragments, links) + Qdrant vectors
    ↓  retrieve / answer
Answers grounded in your own data
```

See [docs/architecture.md](docs/architecture.md) for a detailed breakdown of each layer.

---

## Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- [Docker](https://www.docker.com) (for Qdrant)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set your local model name (default: `gemma4:31b`). See [Configuration](#configuration) for all options.

### 3. Start required services

```bash
# Terminal 1 — Ollama (inference + embeddings)
ollama serve
ollama pull gemma4:31b
ollama pull qwen3-embedding

# Terminal 2 — Qdrant (vector search)
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4. Run the full pipeline on a ChatGPT export

```bash
# Step 1 — ingest, segment, and record in SQLite
python -m jarvis.cli ingest chatgpt --file inbox/ai_chat/chatgpt/raw/<export>.json --persist

# Step 2 — summarize each segment and record in SQLite
python -m jarvis.cli summarize-segments chatgpt --conversation-id <id> --persist

# Step 3 — detect topics and record in SQLite
python -m jarvis.cli detect-topics chatgpt --conversation-id <id> --persist

# Step 4 — extract attributed statements and record in SQLite
python -m jarvis.cli extract-segments chatgpt --conversation-id <id> --persist

# Step 5 — fragment extracts into retrieval units; embed and index in Qdrant
python -m jarvis.cli fragment-extracts chatgpt --conversation-id <id> --persist --embed

# Step 6 — query your data
python -m jarvis.cli retrieve --query "why did we choose Qdrant?"

# Step 7 — ask a question and get a grounded answer
python -m jarvis.cli answer "Why did we choose Qdrant over Pinecone?"
```

---

## CLI Reference

### `ingest chatgpt`

Parse a raw ChatGPT export JSON, normalize it, and split it into segments.

```bash
python -m jarvis.cli ingest chatgpt --file <path-to-export.json> --persist
```

| Flag | Default | Description |
|---|---|---|
| `--file` | *(required)* | Path to the raw ChatGPT export JSON |
| `--output-dir` | `inbox/ai_chat/chatgpt` | Where to write normalized output |
| `--persist` | off | Write source file metadata, conversation, and segment rows to SQLite |

Re-running on an updated export is safe — existing messages are never duplicated. Re-running with `--persist` is idempotent (same SHA-256 → same row, no duplicates).

See [docs/pipeline.md](docs/pipeline.md) for output layout details.

---

### `summarize-segments chatgpt`

Summarize each segment of an ingested conversation using a local LLM, with a rolling context window for continuity.

**Resume-safe:** if a segment summary already exists on disk, the LLM call is skipped. Re-run after an interruption to pick up where you left off.

```bash
python -m jarvis.cli summarize-segments chatgpt --conversation-id <id> --persist
```

| Flag | Default | Description |
|---|---|---|
| `--conversation-id` | *(required)* | Conversation ID (subfolder under `--inbox-dir`) |
| `--inbox-dir` | `inbox/ai_chat/chatgpt` | Base inbox directory |
| `--from-segment` | `0` | Start segment index (inclusive) |
| `--to-segment` | last | End segment index (inclusive) |
| `--context-window` | `3` | Prior segment summaries passed as rolling context |
| `--persist` | off | Write segment summary rows to SQLite (`segment_summaries` table) |
| `--force` | off | Wipe existing files and SQLite records for the range, then re-run |

`--force` with a range only affects that range — other segments are untouched.

---

### `detect-topics chatgpt`

Group segment summaries into topics by measuring cosine similarity between adjacent embeddings. When similarity drops below the threshold, a new topic begins.

```bash
python -m jarvis.cli detect-topics chatgpt --conversation-id <id> --persist
```

Use `--dry-run` to preview topic boundaries without running LLM inference:

```bash
python -m jarvis.cli detect-topics chatgpt --conversation-id <id> --dry-run
```

| Flag | Default | Description |
|---|---|---|
| `--conversation-id` | *(required)* | Conversation ID |
| `--threshold` | `0.55` | Cosine similarity threshold for topic boundaries |
| `--dry-run` | off | Boundary detection only — no LLM calls |
| `--persist` | off | Write topic summary rows to SQLite (`topic_summaries` table) |
| `--force` | off | Wipe existing topic files and SQLite records, then re-run |

Lower threshold → fewer, broader topics. Higher → more, finer-grained topics.

**Prerequisites:** segment summaries must exist (`summarize-segments` must have run first).

---

### `extract-segments chatgpt`

Extract all informational content from each segment as a clean list of attributed statements (speaker + text). This is the first step of the extract→fragment retrieval pipeline.

**Hardened extraction:** segments with embedded code blocks, large prompts, or other structured content are preprocessed deterministically before the LLM call. Risky blocks are replaced with labelled placeholders (`[ARCHIVED_BLOCK_N]`) and an inventory is appended; the raw content is stored in the extract for traceability but never sent to the model. This prevents model drift on code-heavy or prompt-heavy segments.

**3-step retry ladder:** if attempt 1 (inventory placeholders) fails validation, attempt 2 runs archival descriptions of each risky block and retries. If attempt 2 also fails, attempt 3 extracts each message independently and merges results in order.

**Resume-safe:** if an extract already exists for a segment, the LLM call is skipped. Re-run after an interruption to pick up where you left off.

```bash
python -m jarvis.cli extract-segments chatgpt --conversation-id <id> --persist
```

| Flag | Default | Description |
|---|---|---|
| `--conversation-id` | *(required)* | Conversation ID |
| `--inbox-dir` | `inbox/ai_chat/chatgpt` | Base inbox directory |
| `--from-segment` | `0` | Start at this segment index, inclusive |
| `--to-segment` | last | Stop after this segment index, inclusive |
| `--persist` | off | Write extract and statement rows to SQLite (`extracts`, `extract_statements` tables) |
| `--force` | off | Wipe existing extract files and SQLite records, then re-run |

**Prerequisites:** segments must exist (`ingest` must have run first).

---

### `fragment-extracts chatgpt`

Group extracted statements into topically coherent sub-units for semantic retrieval. Each fragment is stored independently and embedded by its raw statement text.

**Resume-safe:** if fragments already exist for a segment, it is skipped unless `--force` is set.

```bash
python -m jarvis.cli fragment-extracts chatgpt --conversation-id <id> --persist --embed
```

| Flag | Default | Description |
|---|---|---|
| `--conversation-id` | *(required)* | Conversation ID |
| `--from-segment` | `0` | Start at this segment index, inclusive |
| `--to-segment` | last | Stop after this segment index, inclusive |
| `--persist` | off | Write fragment and link rows to SQLite (`fragments`, `fragment_statement_links` tables) |
| `--embed` | off | Embed retrieval text and index each fragment in Qdrant (requires `--persist`) |
| `--force` | off | Wipe existing fragment files and SQLite/Qdrant records, then re-run |

`--embed` without `--persist` is rejected with exit code 2.

**Prerequisites:** extracts must exist (`extract-segments` must have run first).

---

### `summarize`

Summarize a single file (conversation, notes) in one shot. Writes disk artifacts only — SQLite persistence for this command will be added in a future release.

```bash
python -m jarvis.cli summarize --file <path>
```

| Flag | Default | Description |
|---|---|---|
| `--file` | *(required)* | Path to `.md`, `.txt`, or `.json` input |
| `--force` | off | Wipe existing output and re-run |

---

### `retrieve`

Query all persisted summaries using natural language. Returns ranked results with source and preview.

```bash
python -m jarvis.cli retrieve --query "what did we decide about the data model?" --top-k 5
```

| Flag | Default | Description |
|---|---|---|
| `--query` | *(required)* | Natural language query |
| `--top-k` | `5` | Number of results to return |

Fragments score higher for specific content and direct statements. Use `--top-k 10` to surface a broader range.

---

### `answer`

Ask a question in plain language and get a grounded answer generated by the local LLM from the top retrieved summaries.

```bash
python -m jarvis.cli answer "Why did we choose Qdrant over Pinecone?" --top-k 5
```

| Flag | Default | Description |
|---|---|---|
| `query` | *(required, positional)* | Natural-language question |
| `--top-k` | `5` | Number of context excerpts to retrieve |
| `--temperature` | `0.3` | LLM sampling temperature |

The answer is grounded in the retrieved summaries — the LLM cites sources where possible. If no relevant context is found, it says so instead of hallucinating.

**Prerequisites:** fragments must be indexed (`fragment-extracts --persist --embed` must have run). Both Ollama and Qdrant must be running.

---

## Configuration

Precedence: **CLI flag > ENV (`.env`) > `config.yaml`**

| Variable | Default | Description |
|---|---|---|
| `JARVIS_PROVIDER` | `local` | `local` \| `openai` \| `benchmark` |
| `LOCAL_MODEL_NAME` | `gemma4:31b` | Ollama model for inference |
| `OPENAI_API_KEY` | — | Required only for `openai`/`benchmark`. Never commit. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cloud model name |
| `JARVIS_OUTPUT_ROOT` | `OUTPUTS` | Root directory for output artifacts |
| `JARVIS_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `qwen3-embedding` | Ollama embedding model |
| `JARVIS_DB_PATH` | `data/jarvis.db` | SQLite database path |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |

---

## Output Layout

```
OUTPUTS/
  <source_basename>/               # single-file summarization
  <conversation_id>/
    segment_summaries/
      <segment_id>.json|.md
    topic_summaries/
      topic_000.json|.md
      topic_001.json|.md
      ...
    extracts/
      extract_000.json|.md
      extract_001.json|.md
      ...
    fragments/
      segment_000/
        fragment_000.json|.md      # segment 0, fragment 0
        fragment_001.json|.md      # segment 0, fragment 1
        ...
      segment_001/
        fragment_000.json|.md
        ...

inbox/ai_chat/chatgpt/
  <conversation_id>/
    normalized.json
    segment_manifest.json
    segments/
      segment_000.json
      ...

data/
  jarvis.db                        # SQLite — relational source of truth (schema v7)
                                   # Tables: source_files, conversations, segments,
                                   #   segment_summaries, extracts, extract_statements,
                                   #   fragments, fragment_statement_links,
                                   #   topic_summaries, topic_segments
```

---

## Error & Exit Policy

- **Hard error** — cannot proceed (missing file, unsupported format, missing API key). Exit code 1.
- **Soft error** — run is degraded but useful (empty input, partial repair, model returned malformed JSON). Output is written with `status: "degraded"` and a `warnings` list. Exit code 0.

---

## Further Reading

- [docs/architecture.md](docs/architecture.md) — system design, storage layers, embedding strategy
- [docs/pipeline.md](docs/pipeline.md) — full ingestion → summarization → topic detection reference
- [docs/OUTPUTS.md](docs/OUTPUTS.md) — output schema spec and field definitions
- [CLAUDE.md](CLAUDE.md) — guidelines for AI-assisted development on this repo
