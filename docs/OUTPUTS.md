# Summarization Output Spec — `jarvis.summarization` v1.0.0

**Purpose:** A single, stable contract for summarization results across providers (local/cloud). This enables CI validation, benchmarking, and consistent human reports.

**Consumers must accept:** any `1.*.*` and **fail** on `>=2.0.0`.

---

## 1) JSON Artifact (machine-readable)

### 1.1 Core (required)
| Field | Type | Constraints | Notes |
|---|---|---|---|
| `summary` | string | ≤ ~200 words (warn at >250) | High-level synthesis; no hallucinations |
| `bullets` | array\<string> | 3–10 items; per item ≤ ~200 chars | Key points / highlights |
| `action_items` | array\<string> | 0–10 items; per item ≤ ~200 chars | Imperative phrasing (“Do X…”) |
| `confidence` | number | 0.0–1.0 | 0.8–1.0 strong; 0.2–0.4 degraded |
| `schema` | string | must equal `jarvis.summarization` | Schema name |
| `schema_version` | string | semver (e.g., `1.0.0`) | See §7 |
| `provider` | string | enum: `local`, `openai`, `benchmark`, … | Actual runner |
| `model` | string | non-empty | e.g., `llama-3-8b-q4`, `gpt-4o-mini` |
| `created_at` | string | ISO-8601 UTC (ends with `Z`) | Generation timestamp |
| `source_file` | string | basename or repo-relative path | Provenance |

### 1.2 Metadata (optional)
| Field | Type | Constraints | Notes |
|---|---|---|---|
| `run_id` | string | timestamp or UUID | Trace multiple outputs |
| `latency_ms` | integer | ≥ 0 | End-to-end latency |
| `token_stats` | object | integers ≥ 0 | `{ input, output, total }` when available |
| `status` | string | enum: `ok`, `degraded` | Mark soft errors |
| `warnings` | array\<string> | free-text | Human-readable caveats |
| `lang` | string | BCP-47 (e.g., `en`, `fr`) | Optional language tag |
| `source_kind` | string | enum: `note`, `conversation` | Helps analytics/UX |

**Strictness:** lenient; unknown extra fields allowed (`additionalProperties: true`).  
**Soft-error signaling:** set `status: "degraded"`, add `warnings[]`, reduce `confidence`.  
**Word/length limits:** guidance only; validators should warn rather than fail.

**Filename convention:**  
- JSON → `OUTPUTS/<timestamp>/<basename>.json`  
- Markdown → `OUTPUTS/<timestamp>/<basename>.md`

---

## 2) Markdown Report (human-readable)

**Section order**
1. `# <Title>` (derived from `source_file`)  
2. `**Confidence:** 0.00` (add a small “Degraded” badge if `status=degraded`)  
3. `## Summary` (≤ ~200 words)  
4. `## Bullets` (unordered list, 3–10 items)  
5. `## Action Items` (checkbox list `- [ ] …`, 0–10 items, imperative tone)  
6. `## Metadata` (2-column table: Provider, Model, Latency (ms), Created At (UTC), Source File, Schema/Version, Language (if present), Run ID (if present))  
7. `## Warnings` (only if `warnings[]` non-empty)

**Rendering rules**
- Keep it factual; no speculation beyond the source.  
- One idea per bullet; group related bullets.  
- Action items must be specific and testable; don’t invent owners/dates unless explicit.  
- Metadata must mirror the JSON artifact exactly.

---

## 3) Error & Exit Policy (summary)

- **Hard error (exit 1):** missing input file; unsupported extension; required secret missing when `provider=openai`; schema validation fails entirely; cannot write outputs.  
- **Soft error (exit 0):** empty/very short input; truncated input; cloud half of benchmark failed; minor output repairs. Mark via `status="degraded"`, add `warnings[]`, lower `confidence`.

---

## 4) Size Guidance for Inputs (Sprint 0)

- **Notes (`.md`)**: Small 700–800 words; Medium 2,000–3,000 words (keep under ~12k tokens).  
- **Conversations (`.json`)**: Short ~30 messages; Medium ~100 messages.

*(Larger tiers are fine later; for now stay within typical local model context.)*

---

## 5) Providers & Outputs

- **Providers:** `local`, `openai`, `benchmark` (local then cloud if key present).  
- **Outputs root:** `OUTPUTS/` with **timestamped** subfolders (`%Y%m%d_%H%M%S`), e.g., `OUTPUTS/20250807_142915/`.

---

## 6) Example Keys (illustrative only)

- **Core:** `summary`, `bullets[]`, `action_items[]`, `confidence`  
- **Metadata:** `schema`, `schema_version`, `provider`, `model`, `created_at`, `source_file`  
- **Optional:** `run_id`, `latency_ms`, `token_stats{input,output,total}`, `status`, `warnings[]`, `lang`, `source_kind`

---

## 7) Versioning Policy

- Start at `schema_version: "1.0.0"`.  
- **Patch:** doc clarifications; no contract change.  
- **Minor:** backward-compatible additions (new optional fields, wider limits).  
- **Major:** breaking changes (rename/remove fields, type changes).  
- Consumers accept any `1.*.*`; they should fail on unknown **major** versions (`>=2.0.0`).

---

## 8) Validation (where to find the schema)

- Formal JSON Schema: `docs/schemas/jarvis.summarization.v1.schema.json` (Draft 2020-12, lenient).  
- Sample files for validators (add in a separate task):  
  - Valid → `docs/schemas/samples/valid/`  
  - Invalid → `docs/schemas/samples/invalid/`

---

## 9) Quality Bars (non-blocking but recommended)

- Summary is faithful and concise; no duplication with bullets.  
- Bullets cover distinct points; avoid restating the summary.  
- Action items are imperative and testable; 0–10 items.  
- `confidence ≥ 0.8` when no warnings; ≤ 0.4 when degraded.  
- Timestamps are UTC ISO-8601; metadata is accurate.

---

## 10) Changelog

- **1.0.0** — Initial public spec: flat core fields; optional `token_stats`, `status`, `warnings`, `lang`, `source_kind`; Markdown layout defined; lenient validation; timestamped output folders.
