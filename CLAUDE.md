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
- Email Embedding Preparation V1: separate backed-up preparation schema, conservative French/English signatures inside quotes, bounded traceable chunks, exact shared content with every occurrence, advisory near matches, resumable profile-scoped CLI, private preview and content-free evaluation. Synthetic validation only; real mom preparation awaits code review.
- Email Processing V1 follow-up: correct CI Flake8 argument order and exclude HTML script/style content from processed text, with synthetic regressions verifying immutable-original recovery. No real profile database access.
- Email Processing V1: deterministic French/English cleaning, conservative recoverable sections, quoted-history retention, separately versioned SQLite representations, verified-backup migration, resumable batches, private comparison and content-free evaluation. Operator subsequently processed all 10,978 mom emails with zero failures; later embedding preparation remains separately reviewed.
- Email ingestion V1: approved at `d0e0f44`. Operator verified the 6.7 GB mom PST: 10,978 emails imported, 101 non-email items skipped, zero failed imports, one metadata warning, archive integrity verified. Immutable original text/metadata remain in SQLite. Ingestion V1 is complete; later processing stays separate.
- PST discovery: standalone read-only inventory and deterministic representative sampling, isolated optional Windows libpff environment, private/structural report separation, and synthetic tests. No email ingestion or real archive inspection; database, vector and web behavior unchanged.
- Phase 3 Commit 3: final synthetic CLI workflows for personal/mom verify full lineage, isolated retrieval/answers, stateful fake Qdrant cleanup/replay, cached artifacts, and legacy UUID reuse. Direct web runners now enforce the existing personal-only guard before SQLite access. Documentation and backup/recovery guidance updated; final approval and real-data deployment remain separate.
- Stabilize the dry-run force regression by comparing SQLite schema/records rather than checkpoint-sensitive file bytes; cover WAL checkpoints and assert cleanup/persistence/vector clients remain unused even with `--persist`.
- Phase 3 Commit 2 review fix: validate cleanup filenames, conversation/segment IDs, indices, and existing SQLite scope before recording or replaying deletion intents. Reject swapped identifiers without mutation; preserve recovery when records or inputs were already removed. Schema v8 and journal format remain unchanged; synthetic tests only.
- Phase 3 Commit 2: scoped delete-first CLI cleanup, dependency invalidation, durable profile recovery journals, atomic validated artifacts, idempotent SQLite repair, and deterministic/reused Qdrant IDs. All processing CLI commands support profiles; serving and web jobs remain personal-only until Phase 4. Schema v8 and existing indexed UUIDs are preserved; synthetic tests only.
- Phase 3 Commit 1 review fix: include normalization staging files in source/output collision preflight; reject raw exports at `normalized.tmp` before any writes and preserve the original source.
- Phase 3 Commit 1: shared profile filesystem boundaries, profile-derived CLI directory defaults, read-only external-source provenance, and non-creating SQLite retrieval/answer access. CLI/web runtime guards remain unchanged; destructive cleanup correctness and recovery are reserved for Commit 2.
- Phase 2 review fix: defer memory-layer Qdrant/embedding clients until first use so SQLite-only CLI and web persistence works without Qdrant; remove the unused topic-detection Qdrant probe. Destructive cleanup and later phases remain deferred.
- Profile Phase 2: profile-aware SQLite/Qdrant clients, centralized client factories, independent embedding Ollama endpoint, and optional remote Qdrant URL/API key. Existing v8 databases validate read-only; new databases keep v8. Non-personal runtime guards remain until pipeline/web isolation. Synthetic tests only; implementation awaits review before deployment or later phases.
- Profile Phase 1: validated `profiles.yaml`, immutable resolved profile context, read-only `config show`, and temporary non-personal CLI/web runtime guards. Review fixes reserve database/sidecar file boundaries and preserve valid YAML merges while rejecting explicit duplicates. Later profile integration remains pending review.
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
- Relational SQLite schema v8 (11 tables: 10 entity tables + `jobs`; deterministic IDs; `INSERT OR IGNORE` idempotency)
- Fragment-only Qdrant index (`jarvis_fragments`; full records reconstructed from SQLite)
- Full ingest → extract → fragment → retrieve pipeline with `--persist` flags throughout
- Web UI V1: read-only operator console (Flask + Jinja2); browsing source → conversation → segment → extract → fragment lineage; ID-first whitelisted file preview; `python -m jarvis.cli serve`
- Web UI upload + ingest: upload ChatGPT exports via browser; background daemon thread runs ingest; `/jobs/<id>` status page with auto-refresh; schema v8 adds `jobs` table
- Web UI extract + fragment jobs: launch extract-segments and fragment-extracts from conversation detail page; reuses same job model; form validation (range, embed→persist, prereq guard); job_detail shows type-specific results with links to extracts/fragments
- Web UI retrieve + answer: Search page (semantic fragment retrieval with scores and links); Answer page (single-turn RAG with clickable citation traceability to fragment→extract→segment); synchronous; reuses `run_retrieval`/`generate_answer` helpers extracted from CLI

### Planned
- Phase 3 final approval and Phases 4–5 remain separately reviewed; profile-bound web/jobs are deferred. Real-data deployment requires verified SQLite/Qdrant/source/artifact/recovery-journal backups; do not use destructive rebuild workflows.
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
| `JARVIS_PROFILE` | `personal` | Global `--profile` takes precedence; CLI processing supports profiles, serving remains personal-only. See `docs/profiles.md`. |
| `JARVIS_PROVIDER` | `local` | `local` \| `openai` \| `benchmark` |
| `LOCAL_MODEL_NAME` | `gemma4:31b` | Ollama model for inference |
| `OPENAI_API_KEY` | — | Required only for `openai`/`benchmark`. Never commit. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Cloud model name |
| `JARVIS_OUTPUT_ROOT` | `OUTPUTS` | Root for all output artifacts |
| `JARVIS_LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `JARVIS_PROMPTS_DIR` | `prompts` | Prompt template directory |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Inference endpoint; default endpoint for embeddings |
| `EMBEDDING_OLLAMA_BASE_URL` | `OLLAMA_BASE_URL` | Optional independent embedding service |
| `EMBEDDING_MODEL` | `qwen3-embedding` | Ollama embedding model |
| `JARVIS_DB_PATH` | `data/jarvis.db` | SQLite database path |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_URL` | — | Optional HTTP(S) endpoint; do not combine with environment host/port |
| `QDRANT_API_KEY` | — | Optional credential; environment/.env only, never display or log |

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
