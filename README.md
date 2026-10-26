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

### PST archive discovery (standalone)

Inspect local Outlook PST archives without Outlook or JARVIS services. This is discovery only;
it does not import mail. Install the optional parser in a separate environment, then run:

```powershell
.venv-pst\Scripts\python.exe tools/inspect_pst.py --pst "D:\Archives\mail.pst" --output-dir "D:\PST_Analysis\quick" --max-samples 150
```

Quick mode inspects up to 3,000 randomly distributed items. Use `--mode full` for all accessible
folder items, `--max-items` to adjust quick mode, `--max-samples 0` for inventory only, and
`--seed` for reproducible sampling. Output defaults to a new timestamped folder under
`%LOCALAPPDATA%\JARVIS\PST_Analysis` and is always required to be outside Git repositories.
Reports: private `pst_inventory.json`, `pst_inventory.md`, and `pst_samples_private.json`;
structural-only `pst_samples_structure.json`. Existing reports are never overwritten.

See [Windows installation, properties, limits and report semantics](docs/pst-discovery.md).

### Email ingestion V1 (PST to SQLite)

Email import uses the selected profile's SQLite database and preserves original text and metadata.
It requires an explicit, verified-backup migration; it never runs a migration at startup.
Install the optional parser in a separate environment and follow the
[safe 100-item trial procedure](docs/email-ingestion.md#safe-100-item-trial).

```powershell
python -m jarvis.cli --profile mom ingest pst --pst "D:\Archives\mail.pst" --limit 100 --dry-run
python -m jarvis.cli --profile mom email migrate --backup "D:\JARVIS_Backups\mom-before-email.sqlite"
python -m jarvis.cli --profile mom ingest pst --pst "D:\Archives\mail.pst" --limit 100
python -m jarvis.cli --profile mom email status
python -m jarvis.cli --profile mom email list --limit 5
python -m jarvis.cli --profile mom email show "<email_id>"
```

`email init` explicitly creates a missing v8 profile database when needed; it refuses existing files.
`--limit` caps new/retried source-item attempts, including non-email items and folder failures.
Previously recorded imports do not consume it. `--dry-run` writes no import records or schema.
`email list --failures` exposes retryable failure IDs and codes; `email show` prints private content.
Omit `--limit` only after reviewing the trial. No LLM, embeddings, cleaning, threading or web changes
are included. See [email storage, identity and recovery details](docs/email-ingestion.md).

### Email Processing V1 (deterministic cleaning)

Clean stored originals in the selected profile without PST access or running services. This
separate layer preserves originals, quoted history and uncertain text, and retains extracted
sections. Install its extension through a verified backup before processing. After review and
explicit authorization for the selected real profile:

```powershell
python -m jarvis.cli --profile mom email processing-migrate --backup "D:\JARVIS_Backups\mom-before-processing-v1.sqlite"
python -m jarvis.cli --profile mom email process --limit 100
python -m jarvis.cli --profile mom email evaluate
python -m jarvis.cli --profile mom email compare "<email_id>"
```

`process` defaults to 100 attempts and resumes by skipping successful current-version records.
`--force` explicitly reprocesses that version within the limit; each forced invocation starts at
the first eligible ID. `evaluate` prints content-free cumulative statistics; `compare` prints
private original/processed text and recoverable sections. There are no embeddings, LLM calls,
thread reconstruction, attachment or web changes. See [cleaning rules, limitations and small mom
evaluation instructions](docs/email-processing.md). Development uses synthetic databases only.

### Email embedding preparation

Prepare versioned, traceable chunks from processed emails with conservative French/English
signature exclusion, exact duplicate references and advisory near-duplicate decisions. This stage
uses SQLite only, with no model calls or vector writes. After code review and operator approval:

```powershell
python -m jarvis.cli --profile mom email preparation-migrate --backup "D:\JARVIS_Backups\mom-before-preparation-v1.sqlite"
python -m jarvis.cli --profile mom email prepare --limit 25
python -m jarvis.cli --profile mom email evaluate-chunks
python -m jarvis.cli --profile mom email preview-chunks "<email_id>"
```

Repeating `prepare` resumes the next bounded batch (default 100 attempts). `--max-chars` (default
1600) and `--overlap` (default 160) select an independent configuration; pass the same values to
preview/evaluation. Preview prints private original, cleaned and final text with decisions.
Original and existing processed records remain unchanged. See [rules, storage, limitations and
small-sample evaluation](docs/email-embedding-preparation.md) before using real data.

### Email signature benchmark (research only)

Compare the unchanged V1 detector with optional local Melusine, Mail Reply Parser, Talon and
conservative repetition/agreement prototypes. Install optional dependencies in `.venv-signature`
with `tools/requirements-signature.txt`, then run from the repository root:

```powershell
$env:PYTHONPATH = "src;."
.venv-signature\Scripts\python.exe -m tools.email_signature_benchmark run --input tests/fixtures/signature_benchmark.json --reference tests/fixtures/signature_reference.json --detectors baseline mail-reply melusine talon-brute talon-ml corpus hybrid baseline-hybrid --report signature-results-new.json
```

`--report` writes fresh content-free metrics. Optional `--private-review` writes full source text,
labels and detector spans to a fresh path outside Git. No LLMs, vectors, PST or production writes
occur. Real-data access is a separate, explicitly invoked `export-readonly` command requiring
`--db`, `--profile`, `--allow-private-read` and `--private-output`; `--count` defaults to 75
(50–100), `--seed` to 20261026. Optional `--private-reference` exports the disjoint remaining
corpus for repetition evidence. Do not run it before review. See the
[research, synthetic findings, metrics and private annotation procedure](docs/email-signature-research.md).

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

These commands use `personal` by default. Complete the
[backup checkpoint](docs/profiles.md#mandatory-checkpoint-before-real-data-deployment) before
processing existing real data. For explicit selection, place `--profile personal` or `--profile mom`
before the command; see the [multi-profile workflow](docs/profiles.md#cli-workflow-and-recovery).

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

## Web App

JARVIS includes a browser UI for browsing ingested data and uploading new exports.

### Start the web server

```bash
python -m jarvis.cli serve
```

Serving and web background jobs remain personal-only until Phase 4. Non-personal profiles use
the CLI workflow; no web profile selector or profile-aware background jobs are included.

Then open [http://localhost:5000](http://localhost:5000) in your browser.

The UI lets you:
- View counts and recent records on the **Dashboard**
- Browse **Sources**, **Conversations**, **Segments**, **Extracts**, and **Fragments**
- Follow the full data lineage: source → conversation → segment → extract → fragment
- Inspect raw JSON/text files for each entity (whitelist-restricted)
- **Upload** a ChatGPT export JSON and ingest it without using the CLI
- **Run extraction** for a conversation (extract attributed statements per segment)
- **Run fragmentation** for a conversation (group statements into retrieval units; optionally embed in Qdrant)
- **Search** fragments by semantic query at `/search` — ranked results with scores and lineage links
- **Answer** a question with RAG at `/answer` — LLM-generated answer with clickable citations traceable back to the source segment
- **Track all pipeline jobs** at `/jobs` — status updates live as each job runs

### Upload and ingest via web

1. Open [http://localhost:5000/upload](http://localhost:5000/upload).
2. Select **ChatGPT** as the source type.
3. Choose your export `.json` file (max 50 MB).
4. Click **Upload and ingest** — you are redirected to a live job status page.
5. When the job succeeds, links to the new Conversation and Source appear.

### Extract and fragment via web

1. From any **Conversation detail** page, click **Run extraction →**.
2. Set optional segment range, leave **Persist to DB** checked, click **Run extraction**.
3. The job status page auto-refreshes. On success, links to each extract appear.
4. Back on the Conversation detail page, click **Run fragmentation →**.
5. Leave **Persist to DB** and **Embed in Qdrant** checked (defaults), click **Run fragmentation**.
6. On success, links to each fragment appear; fragments are now searchable via `retrieve` (CLI) or the **Search** page.

> **Note:** Fragmentation requires extracts to exist first. The fragment form shows a warning
> and blocks submission when no extracts are found for the conversation.

### Search and Answer via web

Once fragments are embedded in Qdrant:

1. Open [http://localhost:5000/search](http://localhost:5000/search), enter a query, click **Search**.
2. Results show rank, score, title, snippet, and links to fragment / extract / segment.
3. Open [http://localhost:5000/answer](http://localhost:5000/answer), enter your question, click **Generate Answer**.
4. The generated answer appears above a citations list. Each citation links to the fragment detail page, from where you can navigate to the extract, segment, and full segment text.

> **Note:** Requires Qdrant to be running and fragments to be embedded (`fragment-extracts --embed`).

> **Note:** The server requires a compatible schema v8 database, including `jobs`.
> Older or incomplete databases fail unchanged. Do not delete or rebuild existing data;
> any future upgrade requires a separately approved backup and migration plan.

---

## CLI Reference

### `serve`

Launch the JARVIS web UI.

```bash
python -m jarvis.cli serve [--host 127.0.0.1] [--port 5000] [--debug]
```

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `5000` | Bind port |
| `--debug` | off | Enable Flask debug mode (auto-reload, verbose errors) |

---

### `ingest chatgpt`

Parse a raw ChatGPT export JSON, normalize it, and split it into segments.

```bash
python -m jarvis.cli ingest chatgpt --file <path-to-export.json> --persist
```

| Flag | Default | Description |
|---|---|---|
| `--file` | *(required)* | Path to the raw ChatGPT export JSON |
| `--output-dir` | Active profile's ChatGPT inbox | Destination inside the active inbox root |
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
| `--inbox-dir` | Active profile's ChatGPT inbox | Source directory inside the active inbox root |
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
| `--inbox-dir` | Active profile's ChatGPT inbox | Source directory inside the active inbox root |
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

Profile selection: **`--profile` > process `JARVIS_PROFILE` > `.env` > `personal`**.
Inspect the resolved configuration without creating storage or contacting services:

```bash
python -m jarvis.cli config show
python -m jarvis.cli --profile mom config show
```

Profiles are registered in `profiles.yaml`. Phase 3 supports all CLI processing commands for
`personal`, `mom`, and other configured profiles, including scoped cleanup and interrupted-run
recovery. `serve` and every Flask entrypoint remain personal-only until Phase 4. Personal's
existing database, collection, inbox, outputs, and environment overrides remain in place.
Configuration paths resolve against the repository root, independently of the working directory.
CLI directory defaults now come from the selected profile. Explicit `--inbox-dir` and ingestion
`--output-dir` overrides must stay inside its inbox root; relative overrides retain their existing
working-directory interpretation. Generated artifacts stay in the profile's output root. Unsafe
identifiers, traversal, cross-profile sources, and resolved symlink/junction escapes are rejected.
Explicit external source files remain read-only; normalized data records their original absolute
path and SHA-256, so later processing does not need the source. `retrieve` and `answer` require an
existing compatible database and never initialize missing storage or collections.
CLI cleanup retains durable intents in `<output_root>/.jarvis-recovery/`; fragment directories
also contain `fragment_manifest.json` for interrupted output rendering. Keep these recovery files.
Large extracts additionally retain `.batch_<index>.json` checkpoints for successful model batches.
After interruption, rerun the same command/range without `--force`. See
[Pipeline recovery](docs/pipeline.md#force-and-interrupted-run-recovery) for dependency invalidation
and service requirements. Use one writer per profile; concurrent CLI/web writes are unsupported.
See [Profiles](docs/profiles.md) for validation rules, limitations, and the mandatory backup
checkpoint before deployment against existing personal data.

| Variable | Default | Description |
|---|---|---|
| `JARVIS_PROFILE` | `personal` | Profile selection; overridden by global `--profile` |
| `JARVIS_PROVIDER` | `local` | `local` \| `openai` \| `benchmark` |
| `LOCAL_MODEL_NAME` | `gemma4:31b` | Ollama model for inference |
| `OPENAI_API_KEY` | — | Required only for `openai`/`benchmark`. Never commit. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cloud model name |
| `JARVIS_OUTPUT_ROOT` | `OUTPUTS` | Personal profile's root directory for output artifacts |
| `JARVIS_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama inference server URL |
| `EMBEDDING_OLLAMA_BASE_URL` | `OLLAMA_BASE_URL` | Optional independent Ollama embedding endpoint |
| `EMBEDDING_MODEL` | `qwen3-embedding` | Ollama embedding model |
| `JARVIS_DB_PATH` | `data/jarvis.db` | Personal profile's SQLite database path |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_URL` | — | Optional HTTP(S) URL; omit environment host/port when using it |
| `QDRANT_API_KEY` | — | Optional Qdrant credential; environment or untracked `.env` only |

For distributed services, inference can stay on the desktop while embeddings run on TrueNAS:

```dotenv
OLLAMA_BASE_URL=http://desktop:11434
EMBEDDING_OLLAMA_BASE_URL=http://truenas:11434
QDRANT_URL=https://qdrant.example:6333
```

Keep `EMBEDDING_MODEL=qwen3-embedding` and use the same model on the embedding server.
Changing endpoints does not migrate collections or re-embed data. `QDRANT_URL` accepts
HTTP/HTTPS (including a reverse-proxy path), rejects embedded credentials, queries and
fragments, and overrides YAML host/port defaults. Remove `QDRANT_HOST` and `QDRANT_PORT`
from your environment/`.env` when setting it; supplying both modes is a configuration error.
Within YAML, choose either `qdrant_url` or `qdrant_host`/`qdrant_port`. API keys are omitted
from `config show`; Qdrant service errors do not include raw remote responses.

Existing SQLite v8 databases are validated read-only when opened. Only a missing database
file is initialized. Unsupported, empty, or incomplete existing databases fail unchanged;
there are no schema migrations or automatic rebuilds.

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
  jarvis.db                        # SQLite — relational source of truth (schema v8)
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
