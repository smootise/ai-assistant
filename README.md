# JARVIS — Local-first AI Assistant for PM Workflows

![CI](https://github.com/smootise/ai-assistant/actions/workflows/ci.yml/badge.svg)

JARVIS is a local-first, privacy-friendly assistant. It ingests work data (Slack/Email/Notion), stores embeddings, and lets you run retrieval → summarization → action-item extraction. Hybrid stack: local GPU first, OpenAI fallback. PWA front end.

## Features (MVP scope)
- Ingestion adapters (dummy → real sources)
- Vector DB indexing & semantic search
- Summaries + action-item extraction
- Chat-style UI (PWA)
- CI/CD and modular adapters

## Quickstart (dev)
```bash
# 1) Python venv
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) Run a sanity check
python -m src.main

# 3) Tests & lint
pytest -q
flake8 src
