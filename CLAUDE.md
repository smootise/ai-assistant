# CLAUDE.md — Project Charter & Coding Guardrails

## Purpose
Primary context file for Claude Code. Loaded on every session. Keep it lean — project-wide rules only.
Sub-files carry task-specific detail and are auto-loaded when working in their subtree:
- `src/jarvis/CLAUDE.md` — backend pipeline, modules, design decisions
- `src/jarvis/web/CLAUDE.md` — web UI layer, V1 scope, file preview rules, empty-state contract
- `tests/CLAUDE.md` — mock contracts, fixture conventions

---

## Project Summary

**Name:** JARVIS
**Mission:** Local-first, privacy-friendly AI assistant for PM workflows. Ingest work data, store embeddings, enable retrieval → summarization → action items. Hybrid local/OpenAI stack, local-first in practice.
**Non-Goals (for now):** Fine-tuning, heavy automation workflows, mobile app.

---

## Roadmap

> Update this section at the end of any session where meaningful progress is made.

### Done
- Repo, CI/CD setup
- ChatGPT export ingestion (normalize → segment)
- Segment summarization with rolling context window (resume-safe, `--force` support)
- Topic detection via cosine similarity + LLM topic summarization
- SQLite persistence layer (source of truth, schema v4)
- Qdrant vector index (retrieval embeddings)
- Semantic retrieval CLI (`retrieve` command)
- RAG answer generation (`answer` command — grounded in retrieved fragments)
- Rename refactor: chunk→segment (ingestion unit), segment→topic (thematic grouping)
- Extract pipeline: clean attributed statements per segment (`extract-segments --persist`)
- Fragment pipeline: topically coherent retrieval units (`fragment-extracts --persist --embed`)
- Relational SQLite schema v7 (10 tables; deterministic IDs; `INSERT OR IGNORE` idempotency)
- Fragment-only Qdrant index (`jarvis_fragments`; full records reconstructed from SQLite)
- Full ingest → extract → fragment → retrieve pipeline with `--persist` flags throughout
- Web UI V1: read-only operator console (Flask + Jinja2); browsing source → conversation → segment → extract → fragment lineage; ID-first whitelisted file preview; `python -m jarvis.cli serve`

### Planned
- Web UI V2: uploads, job launching, retrieve/answer with citations
- Token-budget retrieval: replace flat `--top-k` with `--max-context-tokens` to handle variable fragment sizes
- `summarize` (single-file) `--persist`: wire standalone files through the same ingest→extract→fragment pipeline
- TrueNAS deployment: migrate JARVIS services (Ollama, Qdrant, backend) to run on TrueNAS
- API integrations: Notion, Slack, and other data sources as ingestion adapters
- Logging overhaul: structured log file output (errors + raw model responses on parse failure written to file, not terminal), log rotation, configurable verbosity per module

---

## Coding Standards (Python)

- **PEP8** compliant; **Black-style** line length **100**.
- Avoid **flake8 E/F** errors. Top-level definitions separated by 2 blank lines.
- **Type hints** on all new/modified functions.
- **Logging:** use `logging` module for diagnostic output. Use `print` only for intentional CLI-facing output (result tables, progress reports, dry-run plans).
- **Tests:** Pytest. Add tests for new modules and non-trivial logic.
- **Dependencies:** add to `requirements.txt`; pin when stable.
- No comments unless the *why* is non-obvious. No docstrings restating what the name says.

---

## Git & CI

- **Conventional Commits:** `feat:`, `fix:`, `chore:`, `docs:`.
- **CI:** lint + tests run on every push/PR via GitHub Actions. All PRs must pass.

---

## Configuration

Precedence: **CLI flag > ENV / `.env` > `config.yaml`**

| Variable | Default | Notes |
|---|---|---|
| `JARVIS_PROVIDER` | `local` | `local` \| `openai` \| `benchmark` |
| `LOCAL_MODEL_NAME` | `gemma4:31b` | Ollama model for inference |
| `OPENAI_API_KEY` | — | Required only for `openai`/`benchmark`. Never commit. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cloud model name |
| `JARVIS_OUTPUT_ROOT` | `OUTPUTS` | Root for all output artifacts |
| `JARVIS_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `JARVIS_PROMPTS_DIR` | `prompts` | Prompt template directory |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Shared by inference and embedding clients |
| `EMBEDDING_MODEL` | `qwen3-embedding` | Ollama embedding model |
| `JARVIS_DB_PATH` | `data/jarvis.db` | SQLite database path |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |

`.env` is gitignored. Use `.env.example` for placeholders.

---

## Documentation Rules

- **Update `README.md`** whenever a new CLI command, ENV var, flag, or output path is added or changed. README is the user-facing reference; keep it in sync.
- **Update `docs/pipeline.md`** when the ingestion, summarization, or segmentation pipeline changes (new steps, flags, output fields, resume/force behavior).
- **Update `docs/architecture.md`** when storage layers, embedding strategy, models, or data flow change.
- **Update `docs/webapp.md`** when web routes, pages, file preview rules, or the extension guide change.
- Do not add per-command deep-dives or pipeline detail back into README — those belong in `docs/`.
- CLAUDE.md files are for Claude. README and `docs/` are for humans.
- Update the **Roadmap** section above at the end of sessions where meaningful progress is made.
