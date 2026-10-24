# Email ingestion V1

This milestone adds original PST email storage and local inspection to JARVIS. It stops before
cleaning, AI processing, thread reconstruction, embeddings, attachment extraction or web changes.
Implementation and tests use synthetic data only. No real archive import or real database
migration was performed during development. Review the implementation before the trial below.

## Installation and profile boundaries

Use a separate email environment to keep the native parser out of the normal JARVIS environment:

```powershell
python -m venv .venv-email
.\.venv-email\Scripts\python.exe -m pip install --only-binary=:all: -r tools/requirements-pst.txt
.\.venv-email\Scripts\python.exe -m pip install -e .
```

Alternatively `pip install -e ".[pst]"` explicitly selects the optional extra. The default project
dependencies and production requirements do not install libpff. The parser is the official pinned
`libpff-python==20260926` used by discovery, importing as `pypff`; do not also install
`libpff-python-windows` in this environment. See [parser compatibility](pst-discovery.md#windows-11-installation).
No Outlook or running services are needed. Email status/show/migrate commands do not load pypff.

Global `--profile mom` selects the resolved mom SQLite path; there is no email-specific database
override. Personal and other profiles keep their configured resources. Sources in another
profile's managed storage or database/sidecars are rejected by the existing path boundary.
Use any configured profile instead of mom to reuse the importer. No inbox/OUTPUTS artifacts,
cleanup journals, ChatGPT source/conversation/segment rows or vector operations are created.

## Schema and explicit migration

Base schema v8 remains unchanged and accepted by existing clients. Email schema version 1 is
an additive extension tracked in `_jarvis_meta` as `email_schema_version` and
`email_profile_name`. It uses three tables:

| Table | Role |
| --- | --- |
| `email_archives` | Full-file SHA-256 identity, byte size, parser version and integrity state |
| `email_import_runs` | Run ID, archive/source path, status, counters, timestamps and fixed error code |
| `email_items` | Source locations; imported original emails; skipped non-email items; retryable message, folder and archive failures |

`email_items.original_body` stores the full extracted original text. Metadata, recipients,
transport headers, parsed reply/reference headers and body provenance have separate JSON columns.
Successful originals cannot be overwritten through import/resume. A future cleaned/AI-ready
representation should reference `email_id` in a separate table, preserving the original. No
cleaned representation is generated in V1.

Existing databases are never initialized, deleted, recreated or silently migrated. `email init`
is an explicit operation only for a genuinely missing profile database; it refuses existing
files and inherits the base store's sidecar safeguards. It creates base v8 only. Imports require
an initialized, explicitly migrated database. `email migrate --backup <new-path>` performs:

1. Read-only base-v8 and extension preflight; refuse incompatible/unversioned partial schemas.
2. Create a new backup file outside Git and managed artifacts using SQLite's online backup API.
   A read snapshot includes committed WAL data; copying the `.db` file alone is not used.
3. Check source and backup integrity/foreign keys and compare logical schema/row fingerprints.
   Close and independently reopen the backup and validate again.
4. Acquire `BEGIN IMMEDIATE` on the selected database and refuse if it changed after backup.
5. Execute additive DDL and version/profile metadata in one transaction. Validate the extension,
   integrity and unchanged pre-existing data/schema before commit. Any failure here rolls back.
6. Validate the committed schema and integrity read-only. Keep the verified backup for recovery.

The logical fingerprint includes arbitrary existing tables, BLOBs, indexes, constraints, views,
triggers, metadata and application/user versions; only the new extension objects/markers are
excluded from its before/after comparison. Existing records and base schema metadata stay intact.
No journal mode is changed on the existing database. Existing backup paths are never overwritten.
An already installed matching extension validates and returns without writing another backup.
There is no automatic migration rollback or downgrade command; restore decisions require operator
review. Never replace a database while SQLite clients are open or delete WAL/SHM files.

Use one writer per profile and close the web app/other JARVIS writers before backup, migration or
import. Follow the existing [backup checkpoint](profiles.md#mandatory-checkpoint-before-real-data-deployment)
for other profile resources. The email migration is a new explicit operation, distinct from the
schema-preserving profile rollout. No migration or import is authorized merely by starting JARVIS.
Backups and databases contain private data: use a local, non-synchronized, access-restricted folder.
Do not commit, upload or paste their contents into external services.

SQLite's [online backup API](https://sqlite.org/backup.html) supplies a consistent source snapshot;
Python exposes it as [`Connection.backup`](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup).

## Safe 100-item trial

Run these manually only after code review and approval of the real source and selected profile.
Replace the two example paths. The backup must be a fresh filename outside every Git repository.
These commands have not been executed against real data by the implementation agent.

```powershell
$emailPython = ".\.venv-email\Scripts\python.exe"
$pstPath = "D:\Archives\mail.pst"
$backupPath = "D:\JARVIS_Backups\mom-before-email-v1.sqlite"

& $emailPython -m jarvis.cli --profile mom config show
if ($LASTEXITCODE -ne 0) { throw "Profile validation failed" }

& $emailPython -m jarvis.cli --profile mom ingest pst --pst $pstPath --limit 100 --dry-run
if ($LASTEXITCODE -ne 0) { throw "Review dry-run failures before continuing" }
```

Confirm that config resolves to mom's intended database. If that database does not exist, explicitly
initialize it once; skip this command for an existing database:

```powershell
& $emailPython -m jarvis.cli --profile mom email init
```

Then perform the backed-up migration and limited import:

```powershell
& $emailPython -m jarvis.cli --profile mom email migrate --backup $backupPath
if ($LASTEXITCODE -ne 0) { throw "Migration failed; do not import" }

& $emailPython -m jarvis.cli --profile mom ingest pst --pst $pstPath --limit 100
$trialExitCode = $LASTEXITCODE
& $emailPython -m jarvis.cli --profile mom email status
& $emailPython -m jarvis.cli --profile mom email list --limit 5
& $emailPython -m jarvis.cli --profile mom email list --failures --limit 20
if ($trialExitCode -ne 0) { throw "Review trial status and failure IDs" }
```

Inspect one ID returned by `email list`; this intentionally prints private metadata and body:

```powershell
& $emailPython -m jarvis.cli --profile mom email show "<email_id_from_list>"
```

Stop here for review before any full ingestion. `--limit 100` means at most 100 new/retried source
item attempts, including non-email items and folder errors, so there may be fewer than 100 imported
emails. Previously imported/skipped items do not consume the budget. Repeating this command
advances to another batch after already imported items; it is not a fixed first-100 preview.
Use dry-run to check a resumed batch before authorizing more writes. Dry-run may be used before
initialization/migration, reports `migration_required`, and creates no database, schema, run or
failure rows. It still reads original bodies and performs archive hashing.

## Identity, streaming and recovery

The archive ID is the SHA-256 of all source bytes, computed incrementally. The email ID is a
SHA-256 over that archive ID and an explicit message/folder/index locator. A missing Internet
Message-ID or PST item identifier therefore does not weaken uniqueness. Item identifiers and
Internet Message-IDs are retained separately as evidence, never as uniqueness constraints.
Copies/renames of identical bytes resume the same source. Different archive bytes always get
independent records, even if their Message-IDs match. There is no semantic/cross-archive
deduplication, and multiple source locations with the same Message-ID are preserved.

Folder traversal and body extraction stream one item at a time; memory scales with folder/depth
tracking and a single message, not archive size. No attachment objects/buffers are opened; only
attachment count is stored. Original text is never truncated. A single huge body/transport header
can require substantial memory because libpff returns its value in one call. No 4/14 GB archive
size ceiling is imposed; native memory/runtime and disk capacity remain practical constraints.

The PST has only an `rb` handle, passed to libpff with `mode="r"`. Full hashing before parsing
establishes identity; size, mtime and file identity are checked before each item commit. A full hash
is verified again after a completed/limited scan. Consequently even `--limit 100` reads the entire
6.7 GB archive twice for identity/integrity, in addition to parsing the selected items. Keep the
PST unchanged and close applications that could write it. A source change quarantines its archive
as `source_changed`; committed originals are retained, `show` exposes the quarantine flag, and
further imports/dry-runs of that identity are refused pending operator review. No automatic cleanup
or reassignment is attempted. A caught interruption stops without the final full hash; per-item
stat checks and the initial hash still apply. These checks do not provide a filesystem snapshot
against a malicious concurrent writer or changes restored before verification.

Each item record and its cumulative run counters commit together. Failed reads keep their location,
available metadata and fixed error code. Rerun the same source/profile to retry failed items and
continue past successful ones. No `--force` or deletion path exists. Optional metadata/parser
failures that do not prevent reading the email/body are stored as explicit field states and warning
paths, counted under `warning_messages`; reruns do not overwrite these successful originals.
RTF-only bodies are recorded as failures (`rtf_only_unsupported`) rather than falsely reported as
fully ingested. Messages genuinely lacking both body properties may be imported with a null/empty
original and explicit availability evidence.

Run statuses are `running`, `complete`, `limited`, `partial`, `interrupted`, `failed` and
`source_changed`. `partial` means some item attempts failed. Counts distinguish `imported`,
`would_import` (dry-run), `skipped`, `failed`, `previously_imported`, `previously_skipped`,
`attempted` and `warning_messages`. Folder/archive diagnostics also contribute to failures.
Exit codes: 0 for complete/limited without failed items, 1 for refusal/failure/partial/quarantine,
130 for a caught Ctrl+C. An interruption during initial hashing produces no run because archive
identity is not established yet. A hard kill/power loss may leave a run marked `running`; its
committed counters/items remain the recovery boundary. A later invocation creates a new run and
resumes from the item ledger. Old `running` rows may therefore denote an unclean exit, not a live
process. Disk/storage failures can prevent recording final status; do not infer success from that.
One writer per profile is required; conflicting successful item writes fail instead of overwriting.

## Preserved data and parser limits

The [shared discovery property map](pst-discovery.md#properties-and-their-limits) is reused:
subject; raw/SMTP sender and address type; display To/CC/BCC and available typed recipient rows;
submit/delivery/creation/modification timestamps; Internet Message-ID; Conversation Topic/Index/ID
when available; MAPI In-Reply-To/References; complete available transport headers and separately
parsed reply/reference header values; message class, flags/status, importance/sensitivity and
encoding hints; archive identity/source path, folder components/locator, local item index and
native identifier; attachment count. Sender/recipient Exchange addresses are not resolved by an
external service. Binary metadata is encoded as hex with its byte length in private JSON.

Metadata fields retain `present`, `empty`, `absent`, `unsupported`, and `error` states; no Message-ID,
conversation ID or recipient address is invented. `IPM.Note` and subclasses are email; known
non-email/unsupported classes are explicitly skipped; missing/unreadable classes are failures.
Draft (`UNSENT`) and sent (`FROM_ME`) flags produce `draft` or `sent` evidence; everything else is
`unknown`. A received/delivery timestamp alone never classifies an email as received. Raw flags
and timestamps remain available for a later, separately designed classification strategy.

Plain text is preserved exactly as returned/decoded, including whitespace, quoted replies,
signatures, forwards and marketing text. If missing/empty or unreadable, HTML-to-text fallback
uses the same standard-library converter as discovery. It is not a browser rendering; scripts,
styles and head text are ignored. Body source, decoding, character length and availability/error
states are stored. Byte bodies use BOM/UTF-8, then declared codepage, then explicit replacement
decoding with a warning; the original PST remains the authoritative byte source. RTF extraction,
arbitrary named/multivalue properties, thread reconstruction and address-book resolution are
unsupported. The parser is still libpff alpha; a successful metadata discovery does not prove
every body can be imported. Archive mutation or parser-version changes require review before
resume rather than attempting to reconcile potentially changed identities automatically.

Normal import/status/list output contains counts, IDs and fixed error codes, not subjects, email
addresses, bodies, private paths or native exception text. `email show` explicitly prints the
stored private record. Do not redirect it to a tracked file or share it without manual review.
Original email tables are deliberately absent from existing ChatGPT/web dashboards and retrieval.
No full ingestion or later phase is started by this implementation.
