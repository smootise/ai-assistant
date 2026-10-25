# Email Processing V1 — deterministic cleaning

This is a separate SQLite-only stage after the approved Email Ingestion V1 commit `d0e0f44`.
It reads stored originals, without opening the PST. There are no LLM calls, embeddings, Qdrant
writes, conversation reconstruction, attachment processing, or web changes. Development and
automated evaluation use synthetic temporary databases exclusively. Real processing requires
operator review and explicit authorization.

## Ingestion milestone verified by the operator

The approved importer was successfully run against a **6.7 GB PST** in the **mom** profile:
**10,978 imported emails**, **101 skipped non-email items**, **zero failed imports**, and
**one metadata warning**. Archive integrity was verified. Full original email text and metadata
are stored in that profile's SQLite database. These results were supplied in the milestone
handoff; this implementation did not inspect or modify the real database. Ingestion V1 is
complete; cleaning does not change its parser, identities, import ledger, or stored originals.

## Storage and migration decision

Base schema v8 and email extension v1 remain unchanged. A separately versioned additive
extension, `_jarvis_meta.email_processing_schema_version=1`, installs **one table**:
`email_processed`. The existing email profile marker binds it to the selected profile.
Its composite primary key is `(email_id, processing_version)`; `email_id` references the
immutable `email_items` record. Each row contains:

- Original SHA-256 fingerprint and character length; normalized text and processed text.
- Ordered JSON sections with normalized-text character offsets, full section text, category,
  confidence indicator, detection rule, and inclusion decision.
- Sorted detection/review indicators, a manual-review flag, UTC processing/attempt timestamps,
  and a fixed safe error code when the latest attempt failed.

Algorithm version **`email-clean-v1`** is independent of schema version. Change that constant
when cleaning behavior changes after release. New versions get separate rows; old versions stay
recoverable. `--force` atomically replaces a successful result for the current version without
changing originals. A failed forced attempt retains the previous successful text and timestamp,
marks the attempt failed, and is retried by the next ordinary run. Previous forced revisions
within one algorithm version are not kept as history; the immutable original permits regeneration.

`email processing-migrate --backup <new-path>` reuses the ingestion migration's online SQLite
backup, source/backup integrity and foreign-key verification, logical fingerprint comparison,
independent backup reopening, write lock, intervening-write rejection, transactional DDL, and
post-commit validation. The generalized helper excludes only the extension being installed when
checking preservation: processing migration fingerprints **all** original email tables and rows.
There is no automatic migration, schema repair, downgrade, or database creation. Existing backup
paths, backups in Git repositories, profile database/sidecar collisions, and managed artifact
directories are refused. A matching installed extension validates without creating a new backup.

## Cleaning rules and confidence

`processed_text` is a conservative future retrieval input. It retains quoted information that
may exist nowhere else. It does **not** deduplicate quotations across messages. The categories
are heuristic evidence, not assertions about authorship or reconstructed conversations.

| Rule | Behavior |
| --- | --- |
| CRLF/CR line endings, NBSP/entities, zero-width formatting | Normalize to LF and ordinary spacing; preserve originals |
| Whitespace | Remove trailing whitespace and common outer indentation; reduce runs of blank lines to one blank line; preserve relative list indentation, tabs and column spacing |
| Residual complete HTML blocks | Standard-library conversion with paragraph/list/table boundaries, image alternative text and visible link destinations; flag for review |
| HTML script/style blocks | Exclude their content from normalized/processed text, including standalone blocks; exact content remains recoverable in the immutable original shown by `email compare` |
| Outlook headers | `De`/`From` followed within eight lines by `Envoyé`/`Sent`/`Date` and `Objet`/`Subject` identifies a quoted tail |
| Gmail replies | Single-line `Le ... a écrit :` and `On ... wrote:` identify a quoted tail |
| Forwards | French/English original/forwarded-message delimiters identify a quoted tail |
| `>` quotations | Classify prefixed lines, including multiple nesting levels, as quoted; retain them |
| Standard signature separator | An exact `-- ` line followed by at most eight short name/contact lines before a quote boundary identifies a signature; retain and flag the whole block if it contains currency, decimal prices, numeric dates, product-reference patterns, prose or unrecognized lines |
| Mobile signatures | Recognized French/English iPhone, iPad, Android and Outlook footer lines are excluded only at the end of the message |
| Ambiguous signoffs | `Cordialement`, `Regards`, etc. receive low-confidence signature evidence, stay in output, and require review |
| Nonstandard signature separators | `--` without the exact standard trailing space, or with extra spaces, stays with its following text and requires review |
| Newsletter controls | Exclude only a standalone unsubscribe/browser-view phrase followed by a URL; retain the full offer and other marketing prose |
| Tracking query parameters | Remove `utm_*`, `fbclid`, `gclid`, `mc_cid`, `mc_eid`; preserve destination, fragment and other query parameters |
| Signed URLs | URLs with recognized `sig`, `signature`, `token`, or `x-amz-signature` keys stay unchanged and are flagged |
| Uncertain/malformed input | Retain text and flag incomplete quote markers, unresolved markup/entities, replacement/control characters, malformed links, and empty bodies |

`high`, `low` and `default` are qualitative rule indicators, not calibrated probabilities.
`default` means unclassified text retained as `new`; it does not prove newly authored content.
All quoted sections stay included, even when signatures or newsletter controls occur inside them.
Every normalized character belongs to one section; concatenating section text reconstructs
`normalized_text`. Removed tracking parameters remain in the sections/normalized text. Exact
pre-normalization whitespace and HTML remain recoverable through `email_items.original_body`.

## Batches, recovery and private comparison

All commands use global `--profile`; there is no database override or cross-profile query.
Only imported message rows from verified archives are eligible. Quarantined archives, import
failures and skipped items are not processed. Missing/incompatible databases fail unchanged.
Use one writer per profile and close the web app and other writers before migration/processing.

```powershell
python -m jarvis.cli --profile <profile> email process --limit 100
python -m jarvis.cli --profile <profile> email evaluate
python -m jarvis.cli --profile <profile> email compare "<email_id>"
```

`process` defaults to **100 attempts**, in stable email-ID order, streaming one body at a time.
Each result/error commits independently. Current-version successful rows are skipped without
consuming the limit; failures are retried. Rerun the same command after Ctrl+C or a hard stop.
During ordinary processing an unfinished item has no completion marker and is eligible again.
There are no in-progress
run markers or destructive cleanup journals to repair. Storage errors abort rather than being
reported as successful cleaning; committed earlier results remain available.

`--force` explicitly includes already successful current-version rows, still bounded by `--limit`.
Each forced invocation starts again from the lowest eligible ID; repeated bounded forced commands
revisit that same range. Use a sufficient explicit limit for an intentional full regeneration, or
bump the algorithm version after a reviewed rule change and use ordinary resumable batches.
Never force a real profile merely to try the command.

Progress goes to stderr every 25 attempts and at completion; final JSON reports attempted,
processed, failed, review and already-processed counts. `already_processed` is the eligible
successful count at invocation start, including rows selected again under force. Status `limited`
means the attempt budget was reached (there may be no remaining work); `complete` means the
candidate iterator ended before the limit. Exit codes: **0** complete/limited, **1** partial or
refused/failed operation, **130** caught interruption. Exceptions expose fixed diagnostics only.

`evaluate` is read-only and reports current-version successful result count, average original and
processed character lengths, category percentages weighted by **normalized section characters**,
manual-review count and rows whose latest attempt failed. Percentages include excluded sections
and quoted content; they are classification coverage, not removal percentages. Empty totals
produce zeros. A failed force can contribute its retained previous success and a failed-attempt
count. Manual review includes failed rows. Results are cumulative for this profile/version and
may include records whose archive was quarantined after processing. Statistics print no bodies,
subjects, addresses, paths or native exception text.

`compare` explicitly prints **private content**: original, normalized and processed text plus all
sections, indicators and completion/error state for the selected email/current version. Use the
IDs from `email list`; comparison is read-only. Do not redirect private output into tracked files
or share it without review.

## Small mom evaluation — only after review and explicit authorization

These are operator instructions, **not commands run during development**. Confirm the selected
profile and obtain a new backup filename in a private local folder outside Git. Processing uses
only SQLite, so the regular JARVIS environment suffices; libpff and the PST are unnecessary.
The original email migration is already installed for the imported archive; do not reinitialize.

```powershell
$emailPython = ".\.venv\Scripts\python.exe"
& $emailPython -m jarvis.cli --profile mom config show
# Verify mom's intended database and close other writers before continuing.
& $emailPython -m jarvis.cli --profile mom email processing-migrate --backup "D:\JARVIS_Backups\mom-before-processing-v1.sqlite"
if ($LASTEXITCODE -ne 0) { throw "Processing migration failed; stop" }
& $emailPython -m jarvis.cli --profile mom email process --limit 100
if ($LASTEXITCODE -ne 0) { throw "Processing failed/interrupted; review before continuing" }
& $emailPython -m jarvis.cli --profile mom email evaluate
& $emailPython -m jarvis.cli --profile mom email list --limit 5
& $emailPython -m jarvis.cli --profile mom email compare "<email_id_from_list>"
```

Stop after reviewing this batch and its private comparisons. Do not start processing all 10,978
emails without authorization. A repeated ordinary `process --limit 100` advances to another batch;
it is not a fixed first-100 preview. Evaluation reports whatever has already been processed for
the current version, so its count can exceed 100 after multiple runs.

## Limitations and validation

Rules intentionally retain much newsletter prose, legal disclaimers, long/unmarked signatures,
wrapped/unknown reply markers and ambiguous formatting. A recognized quoted tail may include
interleaved newly authored replies; all of it stays in processed text. Unmarked quoted text can
remain in `new`. HTML conversion is not browser rendering: links, tables and other hidden content
can need manual review. Script/style content is excluded and stays recoverable in the original.
Original HTML fallback information already lost by the importer's text extraction cannot be
recovered without the PST; this stage never reopens it.
Recognized tracking parameters are removed without resolving redirects or fetching links. Unknown
tracking and signed URL conventions remain; signed URLs are recognized only by the listed keys.
An explicit signature separator is still heuristic: short title-cased prose without recognized
valuable patterns can resemble a name. Compare a sample before wider processing; excluded
sections always remain recoverable. Null and empty originals have the same UTF-8 body hash;
their availability distinction remains preserved in the original record.

Memory is bounded by a single message/result, not the archive; extremely large individual bodies
still require memory and duplicate stored derived text. Section JSON also stores normalized text
for straightforward recovery. This favors auditability over minimum disk usage. No schema-driven
thread reconstruction or embedding policy is implied by these categories.

Synthetic tests cover short email, French/English replies and forwards, nested/inline quotations,
signatures, newsletters/URLs, malformed/empty text, prices/product IDs, deterministic output,
recoverable sections, migration backup/WAL/rollback, idempotency, batch limits, interruption,
failure retry, force preservation, algorithm versions, profile isolation, and service prohibition.
The shared test guards refuse SQLite access outside temporary databases and block live services.

Follow-up validation on Windows/Python 3.14.3: the full repository suite passed with
**929 passed, 4 skipped**, including the script/style and immutable-original recovery regressions.
The focused cleaning/processing suite passed **61 tests**. The exact Flake8 command parsed from
the corrected CI workflow passed, as did Black's 100-column check for modified Python files.
No real database evaluation was performed.
