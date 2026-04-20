# CLAUDE.md — Project Charter & Coding Guardrails

## Purpose
Primary context file for Claude Code. Loaded on every session. Keep it lean — project-wide rules only.
Sub-files carry task-specific detail and are auto-loaded when working in their subtree:
- `src/jarvis/CLAUDE.md` — backend pipeline, modules, design decisions
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
- ChatGPT export ingestion (normalize → chunk)
- Chunk summarization with rolling context window (resume-safe, `--force` support)
- Topic segment detection via cosine similarity + LLM segment summarization
- SQLite persistence layer (source of truth, schema v3)
- Qdrant vector index (retrieval embeddings)
- Semantic retrieval CLI (`retrieve` command)
- RAG answer generation (`answer` command — pipes retrieved summaries as context into local LLM)

### Planned
- Web UI: browser-based interface to query JARVIS and view results
- TrueNAS deployment: migrate JARVIS services (Ollama, Qdrant, backend) to run on TrueNAS
- API integrations: Notion, Slack, and other data sources as ingestion adapters
- README reorganization: split into quickstart + per-domain docs pages (`docs/pipeline.md`, `docs/retrieval.md`, etc.)

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
| `LOCAL_MODEL_NAME` | `mistral:7b-instruct` | Ollama model for inference |
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
- Do not add per-command deep-dives or pipeline detail back into README — those belong in `docs/`.
- CLAUDE.md files are for Claude. README and `docs/` are for humans.
- Update the **Roadmap** section above at the end of sessions where meaningful progress is made.
