Folders
/samples/
  notes/
  conversations/

Naming conventions

Notes: note_<size>_<lang>[_<topic>][_v<ver>].md
Sizes: small (700–800 words), medium (2,000–3,000)
Examples: note_small_en.md, note_medium_fr.md, note_small_jarvis_scope_en.md

Conversations: conv_<size>_<lang>[_<topic>][_v<ver>].json
Sizes: short (~30 msgs), medium (~100 msgs)
Examples: conv_short_fr.json, conv_medium_en_release_planning.json

Formats
Notes: Markdown (.md)
- Optional header comment:
<!-- size: small | topic: jarvis_scope | lang: en | created: 2025-08-07 -->
**Conversations** live in `/samples/conversations/` as JSON arrays.

Per-message shape:
```json
{ "speaker": "pm_alex", "content": "…", "ts": "2025-08-07T09:00:00Z" }
speaker: stable handle (e.g., pm_alex, eng_sam, assistant)
content: message text
ts (optional): ISO-8601 UTC (...Z)


Encoding
UTF-8, LF line endings.

Scope (Sprint 0)

Start with four files:
samples/notes/note_small_en.md
samples/notes/note_medium_en.md
samples/conversations/conv_short_fr.json
samples/conversations/conv_medium_en.json

Optional tokens in names
_topic when you have multiple samples per size (e.g., _jarvis_scope)
_v2 only if maintaining multiple versions in parallel