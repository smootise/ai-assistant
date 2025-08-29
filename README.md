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

## Outputs
Each run writes artifacts to a timestamped folder:
OUTPUTS/
  20250807_142915/
    <basename>.json
    <basename>.md

JSON contract (schema jarvis.summarization v1.0.0)
- summary: string (≤ ~200 words)
- bullets: string[] (3–10)
- action_items: string[] (0–10, imperative)
- confidence: number (0–1)
- provider: string (local|openai|…)
- model: string
- latency_ms: integer
- token_stats?: { input, output, total }
- source_file: string
- created_at: ISO-8601
- schema: "jarvis.summarization"
- schema_version: "1.0.0"
- (optional) status: "ok"|"degraded", warnings: string[]

Markdown report mirrors JSON (Title, Confidence, Summary, Bullets, Action Items, Metadata footer).

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

## Quickstart (config)
# create your .env from the template
Copy-Item .env.example .env
# then edit .env in your editor and fill required values
Common toggles
- Switch provider for a single run via CLI flag (preferred), or set JARVIS_PROVIDER in .env.
- Change outputs root via JARVIS_OUTPUT_ROOT (timestamped subfolders are on by default).

## House Rules for AI Edits
See CLAUDE.md for project charter, guardrails, prompts convention, and the config precedence model (CLI > ENV > YAML). When asking an AI assistant to change code or docs, start with:
“Follow CLAUDE.md. Task: …”