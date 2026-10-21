# Profiles — Phase 1

Phase 1 resolves and validates profiles and provides read-only configuration inspection.
It does not enable non-personal ingestion, processing, retrieval, background jobs, or serving.
Those operations fail before dispatch when a non-personal profile is selected. The Flask app
factory applies the same guard, including the `python -m jarvis.web` entrypoint.

## Inspect configuration

```bash
python -m jarvis.cli config show
python -m jarvis.cli --profile mom config show
```

The global `--profile` argument precedes the command. Selection precedence is explicit CLI
argument, process `JARVIS_PROFILE`, `.env` `JARVIS_PROFILE`, then `personal`. Unknown or empty
selections fail rather than falling back. Configuration errors produce CLI exit status 2.

The command displays the selected profile, resolved storage and prompt/sample paths, collection,
shared service/model settings, output settings, logging level, and profile identity. It uses an
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

In Phase 1, custom personal inboxes and collection names can be inspected but cannot be used
for processing: the legacy runtime still reconstructs those resources. Existing personal
database/output overrides continue to work. Directory-override confinement, pipeline path
guards, external-source provenance, collection routing, SQLite compatibility checks, and
profile-bound web operations remain deferred. Existing `--force` cleanup defects are not fixed
in Phase 1; they require separately reviewable changesets and dedicated regression tests.
PST ingestion and optional source copying are outside this feature.

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

Phase 1 must be reviewed with its test results before subsequent work. Cleanup fixes remain
separate review units; commits and pushes require explicit authorization.
