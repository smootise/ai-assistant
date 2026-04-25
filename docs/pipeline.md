# JARVIS — Pipeline Reference

Full reference for the ingestion → summarization → topic detection pipeline.
For command flags and quickstart examples, see [README.md](../README.md).
For system design, see [architecture.md](architecture.md).

---

## Step 1 — Ingest

### ChatGPT export

Export your conversation from ChatGPT (Settings → Data Controls → Export) and place the
raw JSON in `inbox/ai_chat/chatgpt/raw/`.

```bash
python -m jarvis.cli ingest chatgpt \
  --file inbox/ai_chat/chatgpt/raw/<export>.json \
  --persist
```

What happens:
1. Reconstructs the active conversation branch via `current_node` backward walk
2. Filters to visible `user` / `assistant` text messages (excludes system messages, memory updates, hidden messages)
3. Deduplicates adjacent identical user messages (retry collapses)
4. Normalizes to a canonical schema with stable message IDs
5. Splits into overlapping segments: `user → assistant(s) → user`  
   Segment N ends with the user message that opens segment N+1
6. Trailing unmatched user messages (no reply yet) are saved as `pending_tail`

Re-running on an updated export is safe — messages are merged by ID, never duplicated. `--persist` is idempotent: SHA-256-keyed source file rows are never duplicated on re-run.

**SQLite rows written:** `source_files` (raw export + normalized.json), `conversations`, `segments` (with full `segment_text` for citation rendering).

**Reusable entrypoint:** the ingest orchestration is in `src/jarvis/ingest/pipeline.py:ingest_chatgpt()`. Both the CLI (`cmd_ingest` in `cli.py`) and the web upload flow (`web/ingest_runner.py`) call this function directly — no subprocess. The CLI adds the argparse glue + print output; the web runner adds job status tracking.

### Output layout

```
inbox/ai_chat/chatgpt/<conversation_id>/
  normalized.json            # canonical message list
  segment_manifest.json      # segment count, IDs, pending_tail flag
  segments/
    segment_000.json         # segment_id, segment_index, segment_text, message_ids
    segment_001.json
    ...
    pending_tail.json        # only present if last message has no reply
```

`pending_tail.json` is informational only — it is skipped during summarization.

---

## Step 2 — Summarize Segments

```bash
python -m jarvis.cli summarize-segments chatgpt \
  --conversation-id <id> \
  --persist
```

Each segment is summarized individually. The last 3 segment summaries (configurable via
`--context-window`) are passed as rolling context so the model understands continuity.

**Resume behavior:** if a segment's `.json` already exists in
`OUTPUTS/<id>/segment_summaries/`, the LLM call is skipped and the file is loaded from disk.
Re-run the same command after an interruption — only missing segments are processed.

**Partial runs:** use `--from-segment` / `--to-segment` to summarize a specific range.
Rolling context is pre-seeded from existing summaries before the range start.

**Force re-run:** `--force` wipes output files and SQLite records for the specified
range (or all segments if no range given), then re-summarizes. Segments outside the range are
never touched.

**SQLite rows written (with `--persist`):** `segment_summaries` — one row per segment, keyed on `segment_id`. No Qdrant writes at this stage.

### Output layout

```
OUTPUTS/<conversation_id>/segment_summaries/
  <segment_id>.json    # summary, bullets, action_items, confidence, metadata
  <segment_id>.md      # human-readable report
  ...
```

### What each segment summary contains

| Field | Description |
|---|---|
| `summary` | Paragraph summary of the segment |
| `bullets` | Key decisions and insights as bullet points |
| `action_items` | Tasks identified in the segment |
| `confidence` | Model's self-reported confidence (0.0–1.0) |
| `segment_id` | Stable ID linking back to the source segment |
| `segment_index` | Position in the conversation |
| `parent_conversation_id` | Links to the parent conversation |
| `status` | `ok` or `degraded` |
| `source_kind` | Always `ai_chat_segment` |

---

## Step 3 — Detect Topics

```bash
python -m jarvis.cli detect-topics chatgpt \
  --conversation-id <id> \
  --persist
```

Groups consecutive segment summaries into topics. A new topic begins when the
cosine similarity between adjacent segment summary embeddings drops below `--threshold` (default: 0.55).

**Tuning the threshold:**
- Use `--dry-run` first to see the similarity distribution and proposed boundaries without
  running any LLM inference
- Lower threshold → fewer, broader topics
- Higher threshold → more, finer-grained topics
- Typical healthy distribution: mean ~0.70, genuine topic shifts cluster below 0.55

```bash
# Preview boundaries before committing
python -m jarvis.cli detect-topics chatgpt \
  --conversation-id <id> \
  --dry-run

# Example output:
# Similarity distribution across 104 pairs:
#   min=0.41  max=0.94  mean=0.73
#   Boundaries (< 0.55): after s004 (0.44), after s018 (0.51), ...
#   Topics detected: 12
```

**Force re-run:** `--force` wipes all existing topic files and SQLite records for the
conversation, then re-detects and re-summarizes from scratch.

**Prerequisites:** segment summaries must exist in `OUTPUTS/<id>/segment_summaries/`.
Run `summarize-segments` first.

**SQLite rows written (with `--persist`):** `topic_summaries` + `topic_segments` (segment membership mapping). No Qdrant writes at this stage.

### Output layout

```
OUTPUTS/<conversation_id>/topic_summaries/
  topic_000.json    # summary, bullets, action_items + topic metadata
  topic_000.md
  topic_001.json
  topic_001.md
  ...
```

### What each topic summary contains

All segment summary fields, plus:

| Field | Description |
|---|---|
| `topic_index` | Position in the conversation |
| `topic_segment_range` | e.g. `"s000-s018"` — segments covered by this topic |
| `source_kind` | Always `ai_chat_topic` |

---

## Step 4 — Extract Segments

```bash
python -m jarvis.cli extract-segments chatgpt \
  --conversation-id <id> \
  --persist
```

Extracts all informational content from each segment as a clean list of attributed statements
(`speaker` + `text`). This is the first step of the extract→fragment retrieval pipeline.

### Hardening: three views of a segment

The extractor maintains strict separation between three representations:

| View | Where it lives | Used for |
|---|---|---|
| **Source text** | `inbox/.../segments/segment_NNN.json` | Audit / citation — never changed |
| **Working text** | Built in memory for LLM input | Extraction calls — contains placeholders |
| **Retrieval text** | `OUTPUTS/.../fragments/.../fragment_NNN.json` (`text` field) | Qdrant embedding — no raw blobs |

### Risky block detection

Before any LLM call, the segment text is scanned deterministically for risky embedded blocks:

- **Fenced code blocks** — triple-backtick or tilde fences
- **Prompt-like blocks** — sustained blocks with ≥ 2 prompt-style tokens ("You are", "IMPORTANT:", "Return ONLY", "---USER---", etc.) or high heading density
- **XML-like blocks** — runs of lines where ≥ 60 % match XML tag patterns
- **JSON-schema-like blocks** — blocks with `"type": "object"` or `"properties": {` and ≥ 8 lines
- **Long imperative blocks** — ≥ 15-line runs with ≥ 5 lines starting with imperative verbs

Blocks below 400 chars *and* below 6 lines are not archived.

Each detected block is replaced with a placeholder (`[ARCHIVED_BLOCK_1]`, etc.) in the working
text, and a deterministic inventory is appended:

```
---ARCHIVED BLOCKS---
[ARCHIVED_BLOCK_1] kind=fenced_code speaker=user lines=42 chars=1873
```

### 3-step retry ladder

Attempts always use materially different requests — no re-sending the same request:

| Attempt | Input to LLM | Trigger |
|---|---|---|
| 1 | Working text with placeholders + deterministic inventory | Always |
| 2 | Working text with placeholders + archival block descriptions (one LLM call per block to describe it compactly) | Attempt 1 fails validation |
| 3 | Per-message extraction — each speaker message extracted independently, merged in order | Attempt 2 fails validation |

### Drift detection and validation

Each LLM response is validated before acceptance. A response is rejected (and the next
attempt triggered) if any of these hold:

- Unknown top-level keys (only `statements` is allowed)
- Any statement with `speaker` not in `{"user", "assistant"}`
- Statement count exceeds `max(40, line_count // 2)` — implausibly high
- Any 4-gram appears ≥ 5 times across all statements — repetitive drift
- Zero statements extracted from a non-empty segment (> 200 chars)

After all 3 attempts fail validation, the extract receives `status: "partial"` with warnings.

**SQLite rows written (with `--persist`):** `extracts` (one per segment) + `extract_statements` (one per attributed statement). Skips a segment if its extract row already exists — idempotent on re-run.

### Output layout

```
OUTPUTS/<conversation_id>/extracts/
  extract_000.json    # statements + archived_blocks inventory + extraction_attempt
  extract_000.md
  extract_001.json
  extract_001.md
  ...
```

### What each extract contains

| Field | Description |
|---|---|
| `statements` | List of `{statement_id, statement_index, speaker, text, ...}` — per-statement metadata injected deterministically |
| `archived_blocks` | List of detected risky blocks including `raw_text` for traceability |
| `extraction_attempt` | Which attempt (1/2/3) produced the final output |
| `status` | `ok`, `degraded`, `partial`, or `skipped` |

---

## Step 5 — Fragment Extracts

```bash
python -m jarvis.cli fragment-extracts chatgpt \
  --conversation-id <id> \
  --persist \
  --embed
```

Groups extracted statements into topically coherent sub-units for retrieval. Each fragment
is embedded and stored in Qdrant.

**Retrieval-clean text:** fragment `text` (the field Qdrant embeds) is built from the fragment
title + cleaned statement lines + compact archival notes for any archived block references.
Raw block content (code, prompts, etc.) is never included in the retrieval text.

**Traceability:** each fragment links back to its exact statements in `extract_statements` via
`fragment_statement_links`. The extract record holds the full `archived_blocks` list including
`raw_text` for citation and debugging.

**`--persist` vs `--embed`:**
- `--persist` alone — writes fragment rows and statement links to SQLite. Qdrant is not touched.
- `--persist --embed` — additionally embeds each fragment's retrieval text and upserts to Qdrant. Writes `qdrant_point_id` back to the `fragments` table.
- `--embed` without `--persist` — rejected with exit code 2.

**SQLite rows written (with `--persist`):** `fragments` + `fragment_statement_links`. With `--embed`, also updates `fragments.qdrant_point_id` after successful Qdrant upsert.

**Idempotency:** if a fragment row already exists and has a `qdrant_point_id`, it is skipped entirely. If it exists but has no `qdrant_point_id` and `--embed` is set, only the Qdrant indexing step runs (no duplicate insert).

**Qdrant collection:** `jarvis_fragments`. Payload per point: `{fragment_id, parent_conversation_id, segment_id, conversation_date}`. Full fragment records are always fetched from SQLite.

---

## Retrieval

After indexing fragments, query across all data:

```bash
python -m jarvis.cli retrieve --query "why did we choose SQLite over Postgres?" --top-k 5
```

**Tips:**
- Queries hit fragment-level content — specific decisions, direct quotes, action items
- Use `--top-k 10` for broader coverage
- Queries work in any language — the embedding model is multilingual

---

## Answer Generation (RAG)

`answer` is a retrieval-layer command — it works across all persisted data regardless of source.
It does not re-run any part of the ingestion pipeline.

```bash
python -m jarvis.cli answer "Why did we choose SQLite over Postgres?" --top-k 5
```

What happens:
1. Embeds the question with the embedding model
2. Retrieves the top-k most relevant fragments from Qdrant
3. Fetches full fragment records (text + linked statements + segment context) from SQLite via `get_fragments_with_statements`
4. Builds a numbered context block with source citations
5. Renders the `prompts/answer_question.md` template and calls the local LLM
6. Prints the generated answer to stdout

The LLM is instructed to cite sources and to say so clearly if the context is insufficient —
it will not fabricate information not present in the retrieved excerpts.

**Prerequisites:** fragments must be indexed (`fragment-extracts --persist --embed` must have run). Ollama and Qdrant must both be running.
