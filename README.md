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
LOCAL_MODEL_NAME=mistral:7b-instruct  # default
# or:
LOCAL_MODEL_NAME=llama3:8b            # alternative
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
Outputs are saved to `OUTPUTS/<timestamp>/`:
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

```bash
python -m jarvis.cli summarize \
  --file samples/conversations/conv_short_samples_spec_fr_v2.json \
  --persist
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

## House Rules for AI Edits
See CLAUDE.md for project charter, guardrails, prompts convention, and the config precedence model (CLI > ENV > YAML). When asking an AI assistant to change code or docs, start with:
"Follow CLAUDE.md. Task: …"