# Profiles — Through Phase 3 Commit 2

Phase 1 resolves and validates profiles and provides read-only configuration inspection.
Phase 2 adds profile-aware SQLite/Qdrant clients, shared client factories, and independent
service endpoints. Phase 3 Commit 1 adds filesystem boundaries and profile-derived CLI routing
with scoped cleanup and recovery in Commit 2. All processing CLI commands support the selected
profile. Serving and web background workflows remain personal-only until Phase 4. Both `serve`
and the Flask app factory retain their guards, including the `python -m jarvis.web` entrypoint.

## Inspect configuration

```bash
python -m jarvis.cli config show
python -m jarvis.cli --profile mom config show
```

The global `--profile` argument precedes the command. Selection precedence is explicit CLI
argument, process `JARVIS_PROFILE`, `.env` `JARVIS_PROFILE`, then `personal`. Unknown or empty
selections fail rather than falling back. Configuration errors produce CLI exit status 2.

The command displays the selected profile, resolved storage and prompt/sample paths, collection,
shared service/model settings (including both Ollama endpoints and the Qdrant URL), output
settings, logging level, and profile identity. It uses an
explicit allowlist, omits secrets, and redacts URL user information, queries, and fragments.
It does not create managed directories, open databases, construct service clients, or contact
Ollama/Qdrant. It does not inspect database compatibility or service availability.

## Registry and precedence

`profiles.yaml` lives at the repository root and contains a single `profiles` mapping.
The shipped registry defines:

| Resource | personal | mom |
|---|---|---|
| Database | `data/jarvis.db` | `data/profiles/mom/jarvis.db` |
| Collection | `jarvis_fragments` | `jarvis_mom_fragments` |
| Inbox | `inbox/` | `data/profiles/mom/inbox/` |
| Outputs | `OUTPUTS/` | `data/profiles/mom/OUTPUTS/` |

`personal: {}` inherits existing defaults. Its storage and identity precedence is process
environment, `.env`, personal registry values, then existing `config.yaml` defaults.
`JARVIS_DB_PATH`, `JARVIS_OUTPUT_ROOT`, and `JARVIS_USER_NAME` affect personal only.
Service/model settings remain shared. Loading configuration never changes `os.environ`.

Additional profiles require explicit `db_path`, `qdrant_collection`, `inbox_root`, and
`output_root` values. Optional `user_name` defaults to an empty string. For example:

```yaml
profiles:
  personal: {}
  mom:
    db_path: data/profiles/mom/jarvis.db
    qdrant_collection: jarvis_mom_fragments
    inbox_root: data/profiles/mom/inbox
    output_root: data/profiles/mom/OUTPUTS
    user_name: ""
```

Keep secrets and source content out of this file. New profile data stays under ignored `data/`.
Relative configuration paths resolve against the repository root; absolute overrides remain
absolute. CLI input paths retain their existing behavior in this phase.

## Validation and temporary limitations

Every load validates every effective profile, including personal environment overrides and
inactive profile conflicts. Validation is read-only and completes before processing starts.
It rejects malformed registries, duplicate YAML keys, missing/invalid fields, duplicate
collections, database/sidecar collisions, overlapping managed storage, and invalid path types.
Paths are resolved before comparison, including Windows casing, symlinks/junctions, and existing
database hardlinks. Error messages identify conflicting profiles and resolved paths. YAML anchors
and merges retain standard precedence; repeated explicit keys in a mapping remain invalid.

The database is a file boundary rather than ownership of its parent directory:
`data/jarvis.db` may coexist with `data/profiles/mom/jarvis.db`. A profile's inbox/output trees
must be separate from one another and from its database and SQLite sidecars (`-wal`, `-shm`,
`-journal`). These file locations are reserved even when absent: managed directories cannot
equal, contain, or sit beneath them, and reserved files cannot contain other reserved files.
Windows ambiguous drive-relative, reserved, alternate-stream, and trailing-dot/space paths are
rejected.

Custom personal inboxes and collections are supported by the CLI with the same confinement rules
as other profiles. Existing personal database/output overrides continue to work. Web restrictions
remain unchanged. Commit 2 adds scoped, dependency-aware cleanup and interrupted-run recovery;
it does not provide cross-store atomicity or rollback. PST ingestion and optional source copying
remain outside this feature.

## Phase 2 storage and service construction

`jarvis.clients` builds SQLite, Qdrant, embedding, inference, and memory clients from the supplied
resolved configuration. CLI commands, Flask request stores, ingestion persistence, and background
runners use these factories; they never reload the profile from the environment inside a job.
Legacy personal configuration dictionaries retain their original defaults.
Memory construction captures the supplied configuration and defers Qdrant/embedding clients until
their first use. Summarization, extraction, and fragment persistence without embedding can write
SQLite while Qdrant is unavailable, including web jobs. Explicit vector cleanup is unchanged.

`VectorStore(collection_name=...)` routes collection lookup, dimension validation, lazy creation,
upserts, searches, and point deletions to that collection. The direct-constructor default remains
`jarvis_fragments`. Personal keeps that collection; mom uses `jarvis_mom_fragments`. Payloads remain
unchanged. Commit 2 uses deterministic new point IDs and retains recorded existing UUIDs.
Existing collections are never cleared or recreated.
Deletion waits for Qdrant completion, and service failures propagate with sanitized messages.

`SummaryStore(db_path)` retains schema v8 and its public interface. A missing file is created
exclusively and initialized with the existing DDL. An existing file is opened read-only for
version and structural compatibility checks, without DDL, metadata writes, or journal-mode
changes on that file. Validation checks required tables, columns, primary/unique keys, and
foreign keys against a disposable in-memory v8 reference. Harmless extra tables, indexes,
metadata, and compatible extra columns are accepted. Empty files, invalid SQLite, unsupported
versions, missing/incompatible required structures, and orphaned SQLite sidecars fail unchanged.
Later operations use
`mode=rw` so a removed database is never silently recreated.
Phase 3 adds `SummaryStore(db_path, read_only=True)` and matching client-factory options:
they require an existing compatible v8 file, open subsequent connections with `mode=ro` and
`query_only=ON`, and do not initialize files or directories. CLI retrieval and answer use this mode.
For an existing WAL database, SQLite itself may create `-wal`/`-shm` reader coordination sidecars;
the database contents/schema are not rewritten. No immutable-mode shortcut is used, so reads
continue to respect committed WAL data.

Inference uses `OLLAMA_BASE_URL`; `EMBEDDING_OLLAMA_BASE_URL` optionally selects a separate
embedding server and otherwise defaults to inference. YAML equivalent: `defaults.ollama.embedding_base_url`.
The embedding model is unchanged; endpoint selection never starts re-embedding.

`QDRANT_URL` optionally selects an HTTP(S) endpoint, including a reverse-proxy path.
`QDRANT_API_KEY` comes only from process environment or the untracked `.env` and is excluded
from configuration display and service error output. URLs with credentials, queries, or fragments
are rejected. Do not combine URL and host/port variables in environment/`.env`. An environment
URL overrides YAML host/port defaults; within YAML choose either `defaults.memory.qdrant_url`
or host/port. Without a URL, existing `QDRANT_HOST`/`QDRANT_PORT` behavior is preserved.

Tests use synthetic SQLite files and mocked Qdrant/Ollama services. Suite-wide guards block
unmocked Qdrant construction, outbound socket connections, and SQLite paths outside pytest's
temporary tree; configuration loading uses a synthetic repository rather than the clone's `.env`.

## Phase 3 Commit 1: filesystem routing infrastructure

`jarvis.paths.ProfilePaths` derives all managed roots from the resolved `ProfileContext`.
Shared helpers validate path components, imported conversation/segment identifiers, and existing
JSON artifacts before CLI dispatch into file operations. `summarize`, `ingest`, segment summaries,
topic detection, extraction, and fragmentation use these checks. Retrieval and answer continue
using the selected SQLite database and collection with non-creating database access.

Default `--inbox-dir` and ingestion `--output-dir` values are resolved from the selected profile's
ChatGPT inbox. Explicit overrides may select any directory within that profile's inbox root.
Relative overrides resolve from the working directory, preserving valid personal scripts.
Outputs use the profile's output root. Traversal, unsafe Windows names/streams, and resolved
symlink/junction escapes are rejected. Managed files and external sources with multiple hard
links are rejected because their ownership cannot be established from the path alone.

Explicit `--file` sources may live outside managed storage. They are read-only; database/sidecar
files and other profiles' managed files cannot be used as sources. Generated files cannot overwrite
the original source. Normalized files retain `raw_export_path`, `raw_export_sha256`, and a
`source_provenance` list of original absolute paths and SHA-256 values across imports. Standalone
summaries also record `source_provenance`. Source metadata in SQLite retains its existing path/hash
fields. Generated normalized content and segments support later processing after the source is
unavailable. No original is copied, moved, or deleted.

Only necessary artifact directories are created during file-only processing. SQLite is initialized
only for requested persistence; Qdrant collections remain lazy until indexing. `config show` stays
read-only. Boundary validation runs before destructive cleanup; a legacy fragment `--force`
without persistence and without an existing database fails before cleanup instead of initializing
one. Existing cleanup ordering, scope, and dependent-data invalidation remain unchanged.

At Commit 1, the public CLI and Flask guards were retained: `mom` was inspection-only. Direct
handler/pipeline tests exercise the infrastructure without unlocking the runtime. Commit 2 must
separately fix destructive cleanup and recovery before non-personal CLI execution is enabled.
Commit 2 implements CLI recovery and removes the processing guard as described below.
Web profile routing, PST ingestion, and migrations remain unimplemented.
Checks assume filesystem boundaries remain stable during a command; concurrent replacement of
symlinks/junctions is not supported.

## Phase 3 Commit 2: cleanup, recovery, and CLI rollout

`summarize`, `ingest`, `summarize-segments`, `detect-topics`, `extract-segments`,
`fragment-extracts`, `retrieve`, and `answer` now execute for the selected profile. For example:

```bash
python -m jarvis.cli --profile mom ingest chatgpt --file /path/to/export.json --persist
python -m jarvis.cli --profile mom extract-segments chatgpt --conversation-id <id> --persist
python -m jarvis.cli --profile mom fragment-extracts chatgpt --conversation-id <id> --persist --embed
```

All filesystem, SQLite, and Qdrant operations use that profile's resolved resources. `config show`
remains read-only; retrieval/answer still fail without creating resources on an uninitialized profile.
`--force` has inclusive, validated selection boundaries and automatic dependency invalidation,
even when `--persist` is omitted. Dry-run topic detection never cleans up. Cleanup never creates
a missing database or collection. Rerun an interrupted command without force to complete its
durable cleanup intent and repair valid artifact persistence. See [Pipeline](pipeline.md) and
[Architecture](architecture.md#cleanup-and-recovery) for the dependency graph and recovery protocol.

Schema v8 and existing personal data formats/IDs remain compatible. No automatic re-embedding,
migration, collection reset, or source mutation is performed. New fragment manifests and profile
cleanup journals are required recovery artifacts. Keep them with backups. One writer per profile
is supported; web job cleanup/recovery and multi-profile serving remain deferred to Phase 4.

## Mandatory checkpoint before real-data deployment

Implementation and laptop tests use synthetic sources, temporary databases, and mocked services.
Do not deploy this feature against existing personal data until backups and compatibility
verification have been completed and reviewed. Quiesce writers, then capture and verify:

| Resource | Required recovery evidence |
|---|---|
| SQLite | Consistent backup, integrity check, schema/version, metadata, representative row counts |
| Qdrant | Recoverable `jarvis_fragments` snapshot, collection configuration, point count |
| Sources | Inbox and external-source coverage, original paths, checksum inventory |
| Artifacts | Normalized files, manifests, outputs, intermediate files, checksum inventory |

Verify recovery using isolated restored resources before deployment. Confirm personal resolves
to its original resources. Later compatibility checks must accept valid v8 databases without
changing their schema or metadata, tolerate harmless additional objects, and reject incompatible
databases unchanged. No database or existing collection may be deleted, recreated, or migrated
as part of this feature. Do not use the legacy database-deleting rebuild script for verification.

Each phase must be reviewed with its test results before subsequent work. Cleanup fixes remain
separate review units; commits and pushes require explicit authorization.
