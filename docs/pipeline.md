# JARVIS — Pipeline Reference

Full reference for the ingestion → summarization → segmentation pipeline.
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
5. Splits into overlapping chunks: `user → assistant(s) → user`  
   Chunk N ends with the user message that opens chunk N+1
6. Trailing unmatched user messages (no reply yet) are saved as `pending_tail`

Re-running on an updated export is safe — messages are merged by ID, never duplicated.

### Output layout

```
inbox/ai_chat/chatgpt/<conversation_id>/
  normalized.json          # canonical message list
  chunk_manifest.json      # chunk count, IDs, pending_tail flag
  chunks/
    chunk_000.json         # chunk_id, chunk_index, chunk_text, message_ids
    chunk_001.json
    ...
    pending_tail.json      # only present if last message has no reply
```

`pending_tail.json` is informational only — it is skipped during summarization.

---

## Step 2 — Summarize Chunks

```bash
python -m jarvis.cli summarize-chunks chatgpt \
  --conversation-id <id> \
  --persist
```

Each chunk is summarized individually. The last 3 chunk summaries (configurable via
`--context-window`) are passed as rolling context so the model understands continuity.

**Resume behavior:** if a chunk's `.json` already exists in
`OUTPUTS/<id>/chunk_summaries/`, the LLM call is skipped and the file is loaded from disk.
Re-run the same command after an interruption — only missing chunks are processed.

**Partial runs:** use `--from-chunk` / `--to-chunk` to summarize a specific range.
Rolling context is pre-seeded from existing summaries before the range start.

**Force re-run:** `--force` wipes output files and SQLite/Qdrant records for the specified
range (or all chunks if no range given), then re-summarizes. Chunks outside the range are
never touched.

### Output layout

```
OUTPUTS/<conversation_id>/chunk_summaries/
  <chunk_id>.json    # summary, bullets, action_items, confidence, metadata
  <chunk_id>.md      # human-readable report
  ...
```

### What each chunk summary contains

| Field | Description |
|---|---|
| `summary` | Paragraph summary of the chunk |
| `bullets` | Key decisions and insights as bullet points |
| `action_items` | Tasks identified in the chunk |
| `confidence` | Model's self-reported confidence (0.0–1.0) |
| `chunk_id` | Stable ID linking back to the source chunk |
| `chunk_index` | Position in the conversation |
| `parent_conversation_id` | Links to the parent conversation |
| `status` | `ok` or `degraded` |
| `lang` | Detected language |

---

## Step 3 — Detect Segments

```bash
python -m jarvis.cli detect-segments chatgpt \
  --conversation-id <id> \
  --persist
```

Groups consecutive chunk summaries into topic segments. A new segment begins when the
cosine similarity between adjacent chunk summary embeddings drops below `--threshold` (default: 0.55).

**Tuning the threshold:**
- Use `--dry-run` first to see the similarity distribution and proposed boundaries without
  running any LLM inference
- Lower threshold → fewer, broader segments
- Higher threshold → more, finer-grained segments
- Typical healthy distribution: mean ~0.70, genuine topic shifts cluster below 0.55

```bash
# Preview boundaries before committing
python -m jarvis.cli detect-segments chatgpt \
  --conversation-id <id> \
  --dry-run

# Example output:
# Similarity distribution across 104 pairs:
#   min=0.41  max=0.94  mean=0.73
#   Boundaries (< 0.55): after c004 (0.44), after c018 (0.51), ...
#   Segments detected: 12
```

**Force re-run:** `--force` wipes all existing segment files and records for the conversation,
then re-detects and re-summarizes from scratch.

**Prerequisites:** chunk summaries must exist in `OUTPUTS/<id>/chunk_summaries/`.
Run `summarize-chunks` first.

### Output layout

```
OUTPUTS/<conversation_id>/segment_summaries/
  segment_000.json    # summary, bullets, action_items + segment metadata
  segment_000.md
  segment_001.json
  segment_001.md
  ...
```

### What each segment summary contains

All chunk summary fields, plus:

| Field | Description |
|---|---|
| `segment_index` | Position in the conversation |
| `segment_chunk_range` | e.g. `"c000-c018"` — chunks covered by this segment |
| `source_kind` | Always `ai_chat_segment` |

---

## Retrieval

After persisting chunk and segment summaries, query across all data:

```bash
python -m jarvis.cli retrieve --query "why did we choose SQLite over Postgres?" --top-k 5
```

**Tips:**
- Specific content queries (implementation details, exact decisions) → chunks score highest
- Broad thematic queries (overall topic, high-level decisions) → segments score highest
- Use `--top-k 10` to surface both chunk and segment results in the same query
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

**Prerequisites:** summaries must be persisted (`--persist` on `summarize-chunks` or
`detect-segments`). Ollama and Qdrant must both be running.
