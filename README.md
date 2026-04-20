# JARVIS — Local-first AI Assistant for PM Workflows

![CI](https://github.com/smootise/ai-assistant/actions/workflows/ci.yml/badge.svg)

JARVIS is a local-first, privacy-friendly assistant. It ingests work data (Slack/Email/Notion), stores embeddings, and lets you run retrieval → summarization → action-item extraction. Hybrid stack: local GPU first, OpenAI fallback. PWA front end.

## Features (MVP scope)
- Ingestion adapters (dummy → real sources)
- Vector DB indexing & semantic search
- Summaries + action-item extraction
- Chat-style UI (PWA)
- CI/CD and modular adapters

## Configuration & Provider Switch
We use a hybrid config:
CLI flags (one-off overrides)
ENV via .env (secrets + machine defaults)
config.yaml (structured repo defaults; no secrets)
Precedence: CLI > ENV (.env) > config.yaml

## Primary ENV vars
- JARVIS_PROVIDER — local | openai | benchmark (default local)
- OPENAI_API_KEY — only required for openai/benchmark
- OPENAI_MODEL — default gpt-4o-mini
- LOCAL_MODEL_NAME — set when you pick a local model
- JARVIS_OUTPUT_ROOT — default OUTPUTS
- JARVIS_OUTPUT_TIMESTAMP — true|false (default true)
- JARVIS_OUTPUT_TS_FORMAT — %Y%m%d_%H%M%S
- JARVIS_LOG_LEVEL — DEBUG|INFO|WARNING|ERROR (default INFO)
- JARVIS_PROMPTS_DIR — prompts
- JARVIS_SAMPLES_DIR — samples

## Setup
Copy .env.example → .env and fill values (keep .env untracked).
Keep structured, non-secret defaults in config.yaml.

## Samples
Sprint 0 uses files only:
Notes: /samples/notes/ → note_small.md (~700–800 words), note_medium.md (~2,000–3,000 words)
Conversations: /samples/conversations/ → conv_short.json (~30 msgs), conv_medium.json (~100 msgs)
Supported input types: .md, .txt, .json.

### Output Spec & Schema

Summarization runs produce **JSON + Markdown** artifacts in timestamped folders:
`OUTPUTS/<YYYYMMDD_HHMMSS>/<basename>.json|.md`.

- Full spec: see **[docs/OUTPUTS.md](docs/OUTPUTS.md)** (fields, Markdown layout, error policy).
- JSON Schema (Draft 2020-12, lenient): **[docs/schemas/jarvis.summarization.v1.schema.json](docs/schemas/jarvis.summarization.v1.schema.json)**.
- Versioning: artifacts declare `schema="jarvis.summarization"` and `schema_version="1.0.0"`.  
  Consumers accept any `1.*.*` and should fail on `>=2.0.0`.


## Error & Exit Policy
- Hard error → cannot proceed (e.g., missing input file, unsupported extension, required secret absent in openai mode). Exit code 1.
- Soft error → run still useful but degraded (e.g., empty input, cloud skipped during benchmark, partial repair). Mark JSON with status: "degraded", add warnings, lower confidence, and exit 0.
This keeps CI green for soft issues while preserving artifacts for review.

## Style & CI
- Python: PEP8, “Black-style” line length 100, avoid flake8 E/F.
- CI runs lint + tests on each push/PR (see .github/workflows/ci.yml).
- Add new functions with type hints and a quick pytest when possible.

## Providers
- local — runs with your local model (to be wired in Sprints 0–1).
- openai — uses OPENAI_API_KEY and OPENAI_MODEL.
- benchmark — runs local, then (if key present) cloud; saves both to compare latency/quality.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Configuration
```bash
# Create your .env from the template
cp .env.example .env  # or: Copy-Item .env.example .env (Windows)

# Edit .env and set your local model
# Example:
LOCAL_MODEL_NAME=gemma4:31b  # default
# or:
LOCAL_MODEL_NAME=mistral:7b-instruct            # alternative
```

### 3. Start Ollama
Make sure Ollama is running with your chosen model:
```bash
ollama serve
# In another terminal:
ollama pull mistral:7b-instruct
```

### 4. Run Summarization

**English short conversation:**
```bash
python -m jarvis.cli summarize --file samples/conversations/conv_short_samples_spec_en.json
```

**French short conversation:**
```bash
python -m jarvis.cli summarize --file samples/conversations/conv_short_samples_spec_fr_v2.json
```

**Test fixture (minimal 3-message conversation for CI):**
```bash
python -m jarvis.cli summarize --file tests/fixtures/conv_tiny_test.json
```

### 5. View Results
Outputs are saved to `OUTPUTS/<source_basename>/`:
- `<basename>.json` - Machine-readable artifact with full metadata
- `<basename>.md` - Human-readable Markdown report

### Switching Local Models
Edit `.env` to change the model:
```bash
LOCAL_MODEL_NAME=mistral:7b-instruct  # Mistral 7B (default)
LOCAL_MODEL_NAME=llama3:8b            # Llama 3 8B
LOCAL_MODEL_NAME=phi3:medium          # Phi-3 Medium
```

Make sure the model is pulled in Ollama first:
```bash
ollama pull <model-name>
```

## Common Toggles
- Switch provider for a single run via CLI flag (preferred), or set `JARVIS_PROVIDER` in `.env`.
- Change outputs root via `JARVIS_OUTPUT_ROOT` (timestamped subfolders are on by default).
- Adjust log level with `JARVIS_LOG_LEVEL` (DEBUG, INFO, WARNING, ERROR).

---

## Semantic Memory Layer

### Architecture & Responsibility Split

| Layer | Technology | Role |
|---|---|---|
| **OUTPUTS/** | JSON + Markdown files | Raw artifacts from each summarization run |
| **SQLite** (`data/jarvis.db`) | SQLite | **Source of truth** for all structured summary records |
| **Qdrant** | Qdrant (local) | **Vector retrieval index** — embeddings + payload metadata only |

- SQLite owns the canonical record. Every field from the summarization output is stored there.
- Qdrant stores only the vector and a minimal payload (IDs + filter fields). Full records are always fetched from SQLite.
- OUTPUTS files are the raw artifacts. SQLite stores paths back to them.

### Required Local Services

**Ollama** (already running for summarization) — pull the embedding model:
```bash
ollama pull qwen3-embedding
```

**Qdrant** — run via Docker:
```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Summarize + Persist

Add `--persist` to any summarize command to store the result in SQLite and index it in Qdrant:

```bash
python -m jarvis.cli summarize \
  --file samples/conversations/conv_short_samples_spec_en.json \
  --persist
```

Outputs go to `OUTPUTS/<source_basename>/` (stable path — re-running overwrites cleanly).

**Re-run from scratch** (`--force` wipes existing output files and any SQLite/Qdrant record before re-summarizing):

```bash
python -m jarvis.cli summarize \
  --file samples/conversations/conv_short_samples_spec_en.json \
  --persist --force
```

### Semantic Retrieval

Query across all persisted summaries using natural language:

```bash
python -m jarvis.cli retrieve --query "what did we decide about sample file naming?"
```

```bash
python -m jarvis.cli retrieve --query "action items autour de la structure des fichiers" --top-k 3
```

Output format (stdout):
```
Top 2 result(s) for: "what did we decide about sample file naming?"

────────────────────────────────────────────────────────────────────────
#1  score=0.9231  id=1
    source : conv_short_samples_spec_en.json
    created: 2026-04-08T15:37:45Z
    preview: The team locked in the full /samples spec for Sprint 0…

#2  score=0.8107  id=2
    source : conv_short_samples_spec_fr_v2.json
    created: 2026-04-08T17:15:19Z
    preview: L'équipe a finalisé les spécifications du répertoire /samples…
```

### Memory ENV vars

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Shared by inference and embedding clients |
| `EMBEDDING_MODEL` | `qwen3-embedding` | Ollama embedding model name |
| `JARVIS_DB_PATH` | `data/jarvis.db` | SQLite database path |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |

---

## ChatGPT Ingestion Pipeline

Parse a raw ChatGPT conversation export, normalize it to a canonical schema, and chunk it into summarization-ready pieces.

### How it works

1. Reconstructs the active conversation branch via `current_node` backward walk
2. Filters to visible `user`/`assistant` text messages (excludes system, bio/memory updates, hidden messages)
3. Deduplicates adjacent identical user messages (retry collapses)
4. Normalizes to a canonical JSON schema with stable message IDs
5. Chunks with `user → assistant → user` overlap — chunk N ends with the user message that starts chunk N+1
6. Trailing unmatched user messages (no reply yet) are recorded as `pending_tail` in the manifest

### Ingest a ChatGPT export

```bash
python -m jarvis.cli ingest chatgpt --file inbox/ai_chat/chatgpt/raw/<export>.json
```

Custom output directory (default: `inbox/ai_chat/chatgpt`):

```bash
python -m jarvis.cli ingest chatgpt \
  --file inbox/ai_chat/chatgpt/raw/<export>.json \
  --output-dir inbox/ai_chat/chatgpt
```

Re-running the same command on an updated export is safe — new messages are merged in by ID; existing messages are never duplicated.

### Output layout

```
inbox/ai_chat/chatgpt/<conversation_id>/
  normalized.json          # canonical message list (deduped, ordered)
  chunk_manifest.json      # metadata: chunk count, IDs, pending_tail
  chunks/
    chunk_000.json         # each chunk: message_ids, positions, chunk_text
    chunk_001.json
    ...
    pending_tail.json      # present only if last message has no reply yet
```

### What gets written

**`normalized.json`** — top-level fields: `conversation_id`, `title`, `source_platform`, `source_file`, `imported_at`, `message_count`, `messages[]` (each with `message_id`, `speaker`, `created_at`, `position`, `content`).

**`chunk_manifest.json`** — `conversation_id`, `total_visible_messages`, `chunk_count`, `chunk_ids[]`, `pending_tail` (null or a preview), `chunked_at`.

**`chunks/chunk_NNN.json`** — `chunk_id`, `chunk_index`, `start_position`, `end_position`, `message_ids[]`, `chunk_text` (formatted as `speaker: content` pairs).

---

## Chunk Summarization

Summarize each chunk of an ingested conversation using a local LLM. The last N prior chunk summaries are passed as rolling context so the model understands continuity without re-summarizing earlier segments.

**Resume-safe by default:** if a chunk's `.json` already exists on disk, the LLM call is skipped and the existing file is used. This means an interrupted run can be resumed by simply re-running the same command — only missing chunks will be summarized.

### Summarize all chunks

```bash
python -m jarvis.cli summarize-chunks chatgpt \
  --conversation-id <conversation_id>
```

### Summarize a range (useful for testing)

```bash
python -m jarvis.cli summarize-chunks chatgpt \
  --conversation-id <conversation_id> \
  --from-chunk 0 --to-chunk 4
```

### Summarize and persist to SQLite + Qdrant

```bash
python -m jarvis.cli summarize-chunks chatgpt \
  --conversation-id <conversation_id> \
  --persist
```

Existing chunks are loaded from disk; only new ones hit the LLM. All chunks (new + existing) are persisted at the end.

### Re-run from scratch (`--force`)

Wipes output files and SQLite/Qdrant records for the affected range, then re-summarizes:

```bash
# Force all chunks
python -m jarvis.cli summarize-chunks chatgpt \
  --conversation-id <conversation_id> \
  --persist --force

# Force only a specific range — chunks outside it are untouched
python -m jarvis.cli summarize-chunks chatgpt \
  --conversation-id <conversation_id> \
  --from-chunk 50 --to-chunk 60 --persist --force
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--conversation-id` | *(required)* | Conversation ID (subfolder name under `--inbox-dir`) |
| `--inbox-dir` | `inbox/ai_chat/chatgpt` | Base inbox directory |
| `--from-chunk` | `0` | Start at this chunk index (inclusive) |
| `--to-chunk` | last | Stop after this chunk index (inclusive) |
| `--context-window` | `3` | Number of prior chunk summaries passed as context |
| `--persist` | off | Save summaries to SQLite and index in Qdrant |
| `--force` | off | Wipe existing files and DB/Qdrant records for the range before re-running |

### Output layout

Summaries are written to `OUTPUTS/<conversation_id>/chunk_summaries/`:

```
OUTPUTS/<conversation_id>/chunk_summaries/
  <chunk_id>.json    # summary, bullets, action_items, confidence + chunk metadata
  <chunk_id>.md      # human-readable Markdown report
  ...
```

If using `--from-chunk`, rolling context is pre-seeded from already-written summaries for earlier chunks.

---

## Topic Segment Detection

Group consecutive chunk summaries into topic segments by measuring cosine similarity between adjacent embeddings. When similarity drops below the threshold a new segment begins, preserving the chronological narrative arc (topic re-entries produce new segments).

### Detect and summarize segments

```bash
python -m jarvis.cli detect-segments chatgpt \
  --conversation-id <conversation_id>
```

### Dry run — boundary detection only (no LLM calls)

```bash
python -m jarvis.cli detect-segments chatgpt \
  --conversation-id <conversation_id> \
  --dry-run
```

The dry-run prints which chunks would form which segments (with similarity scores) so you can tune `--threshold` before running inference.

### Adjust the similarity threshold

```bash
python -m jarvis.cli detect-segments chatgpt \
  --conversation-id <conversation_id> \
  --threshold 0.60
```

Lower threshold → fewer segments (wider topics). Higher threshold → more segments (finer-grained).

### Detect and persist to SQLite + Qdrant

```bash
python -m jarvis.cli detect-segments chatgpt \
  --conversation-id <conversation_id> \
  --persist
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--conversation-id` | *(required)* | Conversation ID (used to locate `OUTPUTS/<id>/chunk_summaries/`) |
| `--threshold` | `0.55` | Cosine similarity drop threshold for segment boundaries |
| `--dry-run` | off | Detect boundaries only — skip LLM summarization |
| `--persist` | off | Save segment summaries to SQLite and index in Qdrant |

After detection the command always prints a similarity distribution report:

```
Similarity distribution across 104 pairs:
  min=0.41  max=0.94  mean=0.73
  Boundaries (< 0.65): after 686c5e2b_c004 (0.51), after 686c5e2b_c018 (0.44), ...
  Segments detected: 7
```

### Output layout

Segment summaries are written to `OUTPUTS/<conversation_id>/segment_summaries/`:

```
OUTPUTS/<conversation_id>/segment_summaries/
  segment_000.json    # summary, bullets, action_items, confidence + segment metadata
  segment_000.md      # human-readable Markdown report
  segment_001.json
  segment_001.md
  ...
```

Each segment file carries `segment_index`, `segment_chunk_range` (e.g. `"c000-c018"`), and `parent_conversation_id`.

**Prerequisites:** chunk summaries must exist at `OUTPUTS/<id>/chunk_summaries/` — run `summarize-chunks` first. Ollama must be running for embedding (and for LLM inference when not using `--dry-run`). Qdrant is only required for `--persist` or to reuse existing embeddings.

---

## House Rules for AI Edits
See CLAUDE.md for project charter, guardrails, prompts convention, and the config precedence model (CLI > ENV > YAML). When asking an AI assistant to change code or docs, start with:
"Follow CLAUDE.md. Task: …"