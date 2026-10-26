# Email Embedding Preparation V1

Follow-up: [signature detection research and benchmark](email-signature-research.md) compares
local libraries, corpus evidence and conservative combinations in isolation. It includes synthetic
results and a read-only procedure for later operator evaluation. V1 behavior and existing chunks
remain unchanged; `email-prepare-v2` is a proposal only.

This deterministic SQLite-only stage reads successful `email-clean-v1` representations and
prepares text for future indexing. It makes **no LLM, embedding, Qdrant, PST, attachment, thread
reconstruction, or web calls**. Original and existing processed emails are never updated.
Development and validation use synthetic temporary databases only. The real mom database has
not been opened, migrated, or processed during implementation.

The operator reports 10,978 successfully processed mom emails and zero failures, averaging
2,567 original and 2,385 processed characters; classification coverage was 43.42% new, 56.17%
quoted, 0.34% signatures, and 0.07% boilerplate, with 8,365 review flags. These are supplied
Processing V1 results, **not measurements of this preparation stage**. Remaining business
signatures motivated an additional conservative exclusion pass, including inside quotes.

## Storage and migration

Base schema v8, email extension v1, processing extension v1, and `email-clean-v1` are unchanged.
An independent `_jarvis_meta.email_preparation_schema_version=1` extension adds three tables:

| Table | Role |
| --- | --- |
| `email_preparations` | Per-email/configuration checkpoint, source processing version and fingerprint, complete exclusion/retention decisions, success/attempt UTC timestamps and safe failure code |
| `email_embedding_contents` | One exact text per configuration and SHA-256 content identity; token counts and first/last token anchors for near-candidate lookup |
| `email_embedding_occurrences` | Every chunk occurrence, including duplicates: email, configuration, index, content ID, section, normalized start/end offsets, transformations, hard-split flag and recorded advisory near decisions |

The generation key contains the preparation rules version (`email-prepare-v1`), processing
version, maximum character size and overlap. Changing sizes creates independent representations;
bump the rules version for a reviewed behavior/threshold change after release. Old generations
remain available in SQLite. CLI previews/evaluation select the current rules and the supplied
sizes. There is no automatic migration, forced deletion, vector cleanup, or garbage collection.

`email preparation-migrate --backup <fresh-private-path>` reuses the existing migration helper:
online SQLite backup including committed WAL, source/backup integrity and foreign-key checks,
logical fingerprint equality, independent backup reopening, intervening-write rejection under a
write lock, transactional DDL, preservation verification, and post-commit validation. Preservation
includes **all originals and processed rows**. Processing schema must already be installed.
Partial/incompatible schemas and wrong profile markers fail unchanged. Installed matching schemas
validate without another backup. Backup paths must be new, outside Git and managed artifacts,
and must not collide with any profile database or sidecar.

## Signature and footer decisions

The new pass uses complete normalized sections, including content previously excluded by
processing. It rechecks upstream exclusions rather than trusting them blindly. Each normalized
character remains covered by an ordered preparation section with source-section index, message
index, quote depth, category, start/end offsets, confidence, rule and exclusion decision.
Excluded text is recoverable through those offsets and the unchanged processed/original records.

Only a message-tail contact block is eligible for business-signature exclusion:

- It starts at a recognized French/English signoff, a `--` delimiter, or a name-shaped line.
- Every remaining nonblank line must be a supported name, job title, company, telephone, email,
  postal address or website/social-link format. At most 16 nonblank lines are accepted.
- At least a name and a direct telephone/email contact are required. Unmarked blocks require
  both phone and email plus a recognized title/company, a preceding blank boundary and some prior
  body content. An explicit contact/recipient introduction blocks unmarked exclusion. Standalone
  contact cards and weaker unmarked blocks stay included. Company recognition includes legal
  suffixes and explicit labels.
- Currency, dates, references, business/instruction vocabulary, arbitrary prose, bare numeric IDs,
  unfamiliar layouts, and URLs with business identifiers/query strings prevent whole-block
  exclusion. Phone/address formats are narrowly recognized before the generic number guard.
- Standalone unsubscribe/browser-view controls, two exact environmental footer phrases, and
  recognized mobile signatures at a message end have independent exclusion rules.

French and English reply/forward markers, Outlook header groups, and quote-depth changes provide
local message boundaries. Headers stay as separate retained sections/chunks so sender/date/subject
evidence and identifiers are not discarded. Confident signatures are detected independently in
each message, including `>` and nested quoted history. Uncertain blocks remain, with explicit
`ambiguous_signature_retained` or `upstream_exclusion_reconsidered` decisions.

Confidence labels are heuristic evidence, not probabilities. A title-cased phrase can resemble
a name; this is not a general natural-language recognizer. Unknown titles, one-word brands,
long legal disclaimers and complex mixed-column signatures commonly remain. Repeated company
contact blocks are excluded by the same rules; **repetition alone never licenses excluding prose**.
Newsletter offers, prices and instructions remain; there is no wholesale newsletter removal.
Quoted information is retained unless a specific confident footer/signature rule applies.

## Chunk boundaries and provenance

Defaults are **1,600 characters maximum** and **160 characters overlap**. The supported maximum
is 128..100,000 characters; overlap must be nonnegative and below half the maximum. Character
counts are Python Unicode character counts, not model token budgets.

Chunks never cross an email, upstream section, local message, quote-depth, header/body, or
exclusion boundary. A short retained section becomes one chunk. Longer sections prefer a
paragraph/sentence boundary in the latter half of the window, then a newline/space boundary.
Overlap starts at a word boundary and is **at most** the requested size; it can be smaller or zero.
Normal identifiers stay intact. An unbroken token longer than the available bound is split with
`hard_split=true`; source offsets and all pieces preserve it for review. No source text is
silently truncated. A flag is also possible when a short prefix leaves no safe latter-half break.

Chunk text removes recognized line quote prefixes and outer whitespace, and reuses Processing
V1's tracking-parameter removal while retaining signed URLs and business query parameters.
Numeric comparisons such as `> 100` and `>=250` retain their operator. Transformations are
listed per occurrence. Source offsets are **half-open offsets in normalized text**, before these
transformations, not original-PST byte offsets or offsets into the final chunk text.

Repeated quoted text can share the same content identity as its unquoted copy. Different wrapping,
capitalization, punctuation, meaningful numbers or wording keep separate content. Very short
header/signoff chunks can remain because boundaries and conservative retention take priority.
Unmarked replies and interleaved authorship remain imperfectly classified; local message indices
are evidence of boundaries, not reconstructed conversation IDs.

For future Qdrant work, use `(generation_key, content_id)` within the profile's collection as a
content identity and resolve **all** occurrences in SQLite for citations. Email metadata and
stored headers remain available through each occurrence's email/section/message lineage. No
existing ChatGPT fragment index or retrieval behavior is changed here.

## Exact and near duplicates

Exact duplicates use SHA-256 of final UTF-8 chunk text, followed by equality verification before
sharing stored text. All occurrences keep their own position and email reference. There is no
single representative email that hides later copies. Identical text in different sections or
profiles does not erase provenance; profiles never query each other's database.

Near matches are **advisory only and always retained separately**:

1. Casefold and tokenize words, punctuation and mixed identifiers with a fixed regular expression.
2. Require at least 12 tokens. Candidate texts share their first or last three token hashes and
   fall within a conservative token-count band (0.92..1/0.92 of the source count).
3. Compare up to 200 candidates, ordered by content ID. Report candidate truncation explicitly.
4. Require equal sets of numeric/mixed numeric tokens, email tokens and full URL strings, then
   `difflib.SequenceMatcher(..., autojunk=False).ratio() >= 0.96` on token sequences.

This is a transparent lexical comparison, not semantic equivalence. Matching does not authorize
discarding a changed date, negation, name or instruction. The anchor filter intentionally misses
some paraphrases or edits at both ends. The cap can miss further matches; it never merges them.
Recorded near decisions describe candidates present at preparation time. Read-only preview and
evaluation recompute current candidates, so earlier chunks can see later arrivals. Both the
recorded and current decisions are visible in preview. Current statistics are independent of
batch arrival order for the same stored dataset; recorded candidate lists can differ by order.

## Commands, resume and private preview

All commands use the global `--profile`; there is no database override. Use one writer per
profile and close other writers before migration/preparation. No database is created implicitly.

```powershell
python -m jarvis.cli --profile <profile> email preparation-migrate --backup "D:\JARVIS_Backups\fresh.sqlite"
python -m jarvis.cli --profile <profile> email prepare --limit 25
python -m jarvis.cli --profile <profile> email evaluate-chunks
python -m jarvis.cli --profile <profile> email preview-chunks "<email_id>"
# An independent size configuration; use the same sizes for its preview/evaluation:
python -m jarvis.cli --profile <profile> email prepare --limit 25 --max-chars 1200 --overlap 100
python -m jarvis.cli --profile <profile> email preview-chunks "<email_id>" --max-chars 1200 --overlap 100
python -m jarvis.cli --profile <profile> email evaluate-chunks --max-chars 1200 --overlap 100
```

`prepare` defaults to 100 attempts in stable email-ID order. Only successful current-version
processed messages from verified archives are eligible. Successful unchanged inputs skip without
using the budget. Failures retry; input fingerprints detect upstream representation revisions.
No `--force` is needed. Repeat the same command to resume **the next batch**, not to preview a
fixed first batch. Each email's checkpoint, content and occurrences commit in one transaction;
storage errors abort, and completed earlier emails survive. A failed regeneration retains previous
successful chunks and timestamps, records a fixed `preparation_failed`, and retries next time.
Obsolete unreferenced derived content can remain after a revised input; aggregate counts only
include referenced content. No upstream record is changed and no cleanup of private data occurs.

Progress/final batch output contains counts and configuration only. Exit codes: 0 for complete or
limited, 1 for partial/refused/failed operations, and 130 for caught interruption. `limited` means
the budget was reached, not proof of more remaining work. There is no persistent running marker.

`preview-chunks` intentionally prints **private content**: original, normalized and processed
text, preparation sections including excluded source text, final chunks, transformations, exact
references, and advisory duplicate decisions. Source slices are suppressed if the input changed,
with `stale_input=true`; old final chunks remain visible until retry. Do not redirect this into
Git or share it without manual review. The command is read-only.

`evaluate-chunks` is read-only and content-free. It reports prepared/failure/stale counts; total
occurrences, unique exact contents, duplicate occurrences/rate; unique texts with near matches;
candidate truncations; excluded characters by rule/category; retained ambiguous sections; source,
overlapping chunk and unique-text lengths; hard splits; and estimated reduction versus original
and processed lengths. Rates are fractions (multiply by 100 for percentages). Reductions include
exact sharing, formatting/URL cleanup and overlap, and can be negative. They are **not measured
embedding token/cost savings**. Source-character totals omit stale inputs and reduction estimates
are null while any stale input exists. Chunk totals include retained prior successful results,
including failed latest attempts and subsequently quarantined archives. Near-match counts are
texts with a match, not merged groups or unique pair counts. No private paths, subjects, bodies,
addresses or native exception text are printed by aggregate/batch commands.

## Small real-world evaluation — operator only, after review

These instructions are **not commands executed during development**. After reviewing the commit,
confirm the selected profile and a fresh backup outside Git. Stop other writers. Do not re-import
or reprocess the 10,978 original/processed emails and do not initialize an existing database.

```powershell
$emailPython = ".\.venv\Scripts\python.exe"
& $emailPython -m jarvis.cli --profile mom config show
# Verify the intended mom database and backup destination before this explicit migration.
& $emailPython -m jarvis.cli --profile mom email preparation-migrate --backup "D:\JARVIS_Backups\mom-before-preparation-v1.sqlite"
if ($LASTEXITCODE -ne 0) { throw "Migration failed; stop" }
& $emailPython -m jarvis.cli --profile mom email prepare --limit 25
if ($LASTEXITCODE -ne 0) { throw "Preparation failed/interrupted; stop and review" }
& $emailPython -m jarvis.cli --profile mom email evaluate-chunks
& $emailPython -m jarvis.cli --profile mom email list --limit 25
& $emailPython -m jarvis.cli --profile mom email preview-chunks "<email_id_from_list>"
```

Review several prepared IDs privately, including French/English signatures, forwards, quotes,
prices/order references and newsletters. Check that every excluded block contains only contact
or stock-footer information; inspect all retained ambiguity and hard-split flags. Verify exact
duplicates retain all source references and near matches stay separate. Compare original,
processed and final text and inspect reduction statistics. Stop for review before another batch
or a wider run. The first stable-ID batch is not a representative random sample; obtain approval
before expanding coverage. A listed ID may have no preparation if its processing/archive is
ineligible or the attempt failed; inspect status rather than silently switching profiles.

## Synthetic validation

Tests cover French/English business contacts, postal/social footers, nested quotes and forwarded
chains, newsletters, ambiguous business facts, reinstated upstream exclusions, numeric operators,
tracking/signed links, boundaries/overlap, oversized tokens, exact references, advisory near
matches and caps, version/size isolation, verified WAL backups, transactional migration rollback,
idempotency, failed regeneration, input changes, interruption inside transactions, SQLite failure
recovery, private diagnostics, read-only operations and profile isolation. Repository-wide guards
refuse non-temporary SQLite paths and live services. Real-world signature quality and storage/
runtime at full archive scale remain to be evaluated by the operator after code review.

Local validation on Windows/Python 3.14.3: **991 passed, 4 skipped** in the full repository suite;
the focused preparation suite passed **62 tests**. The CI Flake8 command (including the new test
files), Black's 100-column check on modified Python files, and Git whitespace checks passed.
