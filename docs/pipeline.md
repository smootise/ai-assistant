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
  --file inbox/ai_chat/chatgpt/raw/<export>.json
```

What happens:
1. Reconstructs the active conversation branch via `current_node` backward walk
2. Filters to visible `user` / `assistant` text messages (excludes system messages, memory updates, hidden messages)
3. Deduplicates adjacent identical user messages (retry collapses)
4. Normalizes to a canonical schema with stable message IDs
5. Splits into overlapping segments: `user → assistant(s) → user`  
   Segment N ends with the user message that opens segment N+1
6. Trailing unmatched user messages (no reply yet) are saved as `pending_tail`

Re-running on an updated export is safe — messages are merged by ID, never duplicated.

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

**Force re-run:** `--force` wipes output files and SQLite/Qdrant records for the specified
range (or all segments if no range given), then re-summarizes. Segments outside the range are
never touched.

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

**Force re-run:** `--force` wipes all existing topic files and records for the conversation,
then re-detects and re-summarizes from scratch.

**Prerequisites:** segment summaries must exist in `OUTPUTS/<id>/segment_summaries/`.
Run `summarize-segments` first.

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

## Retrieval

After persisting segment and topic summaries, query across all data:

```bash
python -m jarvis.cli retrieve --query "why did we choose SQLite over Postgres?" --top-k 5
```

**Tips:**
- Specific content queries (implementation details, exact decisions) → segments score highest
- Broad thematic queries (overall topic, high-level decisions) → topics score highest
- Use `--top-k 10` to surface both segment and topic results in the same query
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
2. Retrieves the top-k most relevant summaries from Qdrant
3. Fetches full summary records (text + bullets) from SQLite
4. Builds a numbered context block with source citations
5. Renders the `prompts/answer_question.md` template and calls the local LLM
6. Prints the generated answer to stdout

The LLM is instructed to cite sources and to say so clearly if the context is insufficient —
it will not fabricate information not present in the retrieved excerpts.

**Prerequisites:** summaries must be persisted (`--persist` on `summarize-segments` or
`detect-topics`). Ollama and Qdrant must both be running.
