# JARVIS — Codex Instructions

## Project context

JARVIS is a local-first Python AI assistant using
SQLite, Qdrant, and Ollama.

Follow the project conventions in CLAUDE.md.

## Documentation

Read documentation relevant to the task:
- docs/architecture.md for architecture changes.
- docs/pipeline.md for ingestion changes.
- docs/webapp.md for web interface changes.

Additional instructions:
- src/jarvis/CLAUDE.md for backend changes.
- src/jarvis/web/CLAUDE.md for web changes.
- tests/CLAUDE.md for test changes.

## Coding conventions

- PEP8-compliant Python.
- Black formatting, line length 100.
- Type hints for new or modified functions.
- Pytest for automated tests.
- Maintain backward compatibility.

## Safety

- Never delete or recreate existing user databases.
- Never delete or recreate existing Qdrant collections.
- Never modify real user data during tests.
- Keep configuration secrets and private data out of Git.
- Do not commit or push without explicit approval.
- Ask before making destructive or irreversible changes.