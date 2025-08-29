# CLAUDE.md — Project Charter & Coding Guardrails

## Purpose
This document gives Claude Code the **context** and **rules** to follow when generating or editing code in this repository.

---

## Project Summary
- **Name:** JARVIS
- **Mission:** Local-first, privacy-friendly AI assistant for PM workflows. Ingest work data (Slack/Email/Notion), store embeddings, and enable retrieval → summarization → action items. Hybrid local/OpenAI stack.
- **3–6 Month Goals (roadmap):**
  - Sprint 0: Repo, CI/CD, PWA shell
  - Sprint 1: Dummy ingestion
  - Sprint 2: Data model & persistence
  - Sprint 3: Tagging layer
  - Sprint 4: Retrieval & vector indexing
  - Sprint 5: Summaries & action items
  - Sprint 6+: Chat UI and integrations (Slack, Email, Notion, Calendar)
- **Non-Goals (for now):** Fine-tuning, heavy automation workflows, mobile app.

> If a task conflicts with these goals, ask for clarification before proceeding.

---

## Architecture (initial)
- **Frontend:** PWA (Next.js or SvelteKit) — to be decided in Sprint 0.
- **Services (backend/ops):** Python for ingestion, embeddings, simple APIs.
- **Vector DB:** Local Qdrant initially; may evaluate Pinecone/Weaviate later.
- **Models:** Local-first (embeddings & small LLMs); OpenAI fallback for “hard” tasks.
- **Data sources:** Dummy → Slack, Email, Notion, Calendar (incrementally).
- **Repo layout (target):**.
├─ src/ # Python modules
├─ tests/ # Pytest
├─ web/ # (future) PWA app
├─ scripts/ # one-off utilities
├─ prompts/ # prompt templates
├─ docs/ # design notes & ADRs
└─ .github/workflows/ # CI


---

## Coding Standards (Python)
- **PEP8** compliant; **Black-style** formatting with **line length 100**.
- Avoid **flake8 E/F** errors (top-level defs separated by **2 blank lines**).
- Use **type hints** for new/modified functions. Prefer small, pure functions.
- **Docstrings:** Google-style or numpydoc.
- **Logging:** `logging` module (no stray `print` except in `__main__`).
- **Tests:** Pytest for new functions/modules; keep a fast test loop.
- **Dependencies:** add to `requirements.txt`; pin when stable.

> Golden prompt for code generation:  
> **“Write PEP8-compliant Python, formatted for Black (line length 100). Avoid flake8 errors E/F.”**

---

## Git & CI
- **Conventional Commits:** `feat: …`, `fix: …`, `chore: …`, `docs: …`.
- **Branching:** `feat/<area>-<short-desc>`, `chore/…`, `fix/…`.
- **CI expectations:** All PRs must pass lint/tests via GitHub Actions.
- Optional local hooks later: pre-commit with Black/Ruff.

---

## How to Work With This Repo (for Claude)
When asked to make changes:
1. **Read** relevant files (and `CLAUDE.md`) and restate the plan.
2. **Propose a diff** with minimal changes to meet acceptance criteria.
3. Ensure code **passes flake8 & pytest** (assume CI enforces).
4. Update **README** or `docs/` if you introduce new commands or configs.
5. If uncertain, **ask a clarification question** before large edits.

---

## Sprint 0: Current Tickets
- Initialize repo & CI/CD — ✅ (in progress/complete)
- Select & configure AI coding assistant — ⏳
- Scaffold PWA shell (Next.js or SvelteKit) — 🎯 due Aug 21, 2025

---

## Notes for Later
- Move scope/roadmap from Notion into `/docs/` as an ADR or `SCOPE.md`.
- When ready, add Black/Ruff + pre-commit and update CI accordingly.
