# PST archive discovery

This standalone utility helps decide a future email data model. It never imports JARVIS,
opens SQLite, connects to Qdrant/Ollama, sends data over the network, or changes the source PST.
No real archive was inspected during implementation. Running it against an actual archive is
a separate user-authorized action.

## Windows 11 installation

Parser research checked on 2026-10-24:

| Package | Available Windows wheels | Decision |
| --- | --- | --- |
| [`libpff-python` 20260926](https://pypi.org/project/libpff-python/20260926/) | Official libyal release; CPython 3.10–3.15, including 3.14, with Windows architecture-specific wheels | Preferred; pinned in `tools/requirements-pst.txt` |
| [`libpff-python-windows` 20231205](https://pypi.org/project/libpff-python-windows/20231205/) | Third-party packaging of older libpff; CPython 3.10–3.13, Windows AMD64 | Alternative only in a fresh matching Python environment; unavailable for this machine's 3.14 |

Both packages import as `pypff`; never install them together. Both describe libpff as alpha.
The official 20260926 CPython 3.14 AMD64 wheel was successfully installed and imported locally
on Python 3.14.3; native API names were checked. Actual archive parsing and 14 GB throughput
remain unmeasured. Mock tests exercise traversal and extraction without claiming native-parser
validation of real mail. No dependency was added to JARVIS's production environment.

From the repository root, create an isolated environment (64-bit CPython recommended):

```powershell
python -m venv .venv-pst
.venv-pst\Scripts\python.exe -m pip install --only-binary=:all: -r tools/requirements-pst.txt
.venv-pst\Scripts\python.exe -c "import sys, platform, pypff; print(sys.version); print(platform.machine()); print(pypff.get_version())"
.venv-pst\Scripts\python.exe tools/inspect_pst.py --help
```

The first command requires a working Python on PATH. No Outlook, Microsoft COM, C compiler,
JARVIS installation, or services are required. The script itself uses only the Python standard
library and `pypff`. Synthetic tests do not require `pypff`.

If pip reports no matching distribution, first check interpreter version, architecture and
pip compatibility. Do not silently compile an unreviewed substitute. Prefer a fresh supported
64-bit CPython environment with the official wheel. For an older Python 3.10–3.13 AMD64
environment, the third-party package above is an explicitly documented fallback, but it was
not validated here. Another fallback is running this same tool with official Linux wheels
inside WSL, using mounted Windows source/output paths. Upstream documents
[source builds](https://github.com/libyal/libpff/wiki/Building) if a matching binary is
unavailable. No custom PST parser or Outlook-dependent fallback is implemented.

## Inspect the 6.7 GB archive

Choose an explicit source path and an output directory outside Git repositories on a local,
non-synchronized drive. The utility cannot detect cloud synchronization software. Keep these
outputs away from OneDrive, Dropbox, shared folders and automatic uploads. Files inherit the
destination directory's permissions; no separate encryption or ACL policy is installed.

Start with a quick inspection after authorizing the real source path:

```powershell
.venv-pst\Scripts\python.exe tools/inspect_pst.py --pst "D:\Archives\mail.pst" --output-dir "D:\PST_Analysis\quick" --mode quick --max-items 3000 --max-samples 150 --seed 42
```

Examine the private inventory and structural report locally. If useful, perform a full metadata
inventory in a new output directory:

```powershell
.venv-pst\Scripts\python.exe tools/inspect_pst.py --pst "D:\Archives\mail.pst" --output-dir "D:\PST_Analysis\full" --mode full --max-samples 150 --seed 42
```

For body-free inventory, use `--max-samples 0`. Full mode ignores `--max-items`.
`--max-body-chars 200000` caps stored sample text per email; it does not cap native allocation
when fetching a single body. Set a different cap if needed; truncation and original text length
are recorded. `--progress-every 500` controls count-only metadata progress. Native archive
opening and single slow parser calls may not immediately produce progress or respond to Ctrl+C.

If `--output-dir` is omitted, reports go into a new timestamped directory under
`%LOCALAPPDATA%\JARVIS\PST_Analysis` (or the home directory when LOCALAPPDATA is absent).
The timestamp uses UTC. Paths resolve symlinks/junctions; existing report paths, hardlinks to
the source, and output inside this or another Git repository are rejected. Reports are created
exclusively, never overwritten. Choose a new directory for each run.

## Reports and privacy

| File | Contents | Handling |
| --- | --- | --- |
| `pst_inventory.json` | Archive format/size, private folder names and hierarchy, declared counts, inspected email counts, dates, item classes, property statistics and error counts | Private; contains no bodies or subjects |
| `pst_inventory.md` | Readable folder/count/date/availability summary | Private; contains no bodies |
| `pst_samples_private.json` | Selected actual metadata, headers, recipients, text and attachment metadata | Private; keep local |
| `pst_samples_structure.json` | Availability states, types, lengths, aggregate statistics and structural examples | No email text, subjects, names, addresses, filenames, identifier values, dates or folder names |

The structural report is constructed from an allowlisted schema. Strings/binary values become
types and lengths; dates become types; source numbers become types. No regex redaction or
hashes of private values are used. Static direction/body-format categories and aggregate counts
are retained. Per-sample attachment and recipient fields retain their availability and shape.
Counts and lengths can still reveal archive characteristics; this is a structural disclosure,
not a formal anonymization or differential privacy guarantee. Review before sharing.

Normal console output contains fixed messages and progress counts only. Parser exception text,
subjects, paths, and bodies are never logged. Report filenames and PST/OST archives are also
gitignored as a secondary safeguard. The tool contains no upload capability. Do not commit or
share the three private reports. Manually redacted email examples, if wanted later, are separate.

## Scope, sampling and completeness

The tool first enumerates the reachable folder tree with an iterative depth-first traversal.
Folder locators are child-index sequences; JSON paths are arrays of name components, avoiding
ambiguity from duplicate folder names or slashes. Counts declared by libpff and inspected email
counts are reported separately. They can disagree when there are non-email or malformed items.

Quick mode draws up to `--max-items` unique positions uniformly across the declared folder
item space, then inspects them in folder/index order. It does not inspect only the first items.
All reachable folder names/counts are still enumerated, so quick mode is not a strict time limit.
Small folders may be absent from the limited random subset. Full mode scans every accessible
item exposed by each folder's `get_sub_message` interface. It is a metadata inventory, not a
recovery, orphan, hidden/associated-item or deleted-message forensic scan.

The sampling phase maintains a uniform reservoir plus one random-priority candidate per
observed marginal feature bucket: folder, year, direction, reply/forward heuristic, attachments,
conversation metadata and body format. A greedy coverage pass chooses candidates that cover
the most remaining buckets, then fills remaining slots from the reservoir. It is deterministic
for the same archive, parser version, seed, mode and limits. Rare buckets receive attention,
but small budgets cannot guarantee every bucket or combination. Samples are intentionally
diverse and are not weighted statistical estimates. Fewer than the requested samples are
possible when there are fewer inspected emails or samples fail to read. Quick samples represent
only the inspected subset. Body-format sampling uses property presence, not decoded content.

Memory scales with folders, observed years, property/class counters and the selected samples,
rather than message count or archive byte size. The source is accessed through a Python `rb`
handle passed to `open_file_object(..., mode="r")`; no writable source handle exists. Size and
mtime are checked before/after; this is not a cryptographic source integrity check. Close any
application that might modify the archive concurrently. No artificial 4 GB/14 GB size ceiling
is imposed, but native-parser memory/runtime and individual oversized bodies remain constraints.
Attachment binaries and embedded attached messages are never fetched, and RTF is never
decompressed. A full inventory avoids fetching all three body values: property descriptors
provide presence evidence. libpff may internally read blocks containing those properties.

Status values are `complete`, `limited`, `partial_errors`, `partial_sample_errors`, `interrupted`,
`failed`, or `source_changed_externally`. Even `complete` means all accessible folder items,
not recovered or hidden contents. Partial/error reports preserve counters from completed work;
failed folders/items are counted with fixed error codes. A caught Ctrl+C writes partial reports
and exits 130. Hard process termination, power loss, or another interruption during final report
writing can leave incomplete files; no resume/checkpoint support is promised. Exit 0 means a
successful complete/limited inspection, 1 indicates reported errors/source change, and 2 means
startup/output validation failed. Reports are finalized only at the end or a caught interruption.

## Properties and their limits

Every extracted field uses `{status, value}`. States distinguish `present`, `empty`, `absent`,
`unsupported` (API or property value type not supported), and `error` (parser failed).
No fabricated values replace missing data. After a property-table parsing failure, unobserved
fields are errors/unsupported rather than falsely absent. Availability counts use inspected
emails as their denominator; absent/empty/error/unsupported counts are separate. Numeric
property-ID/value-type pairs are counted without decoding unknown values; missing counts for
these discovered tags can be derived from the inspected email count.

The adapter uses the actual libpff record-set API rather than assuming arbitrary Outlook
attributes exist. Relevant upstream definitions are in
[`pypff_record_entry.c`](https://github.com/libyal/libpff/blob/main/pypff/pypff_record_entry.c),
[`pypff_message.c`](https://github.com/libyal/libpff/blob/main/pypff/pypff_message.c), and
[`mapi.h`](https://github.com/libyal/libpff/blob/main/include/libpff/mapi.h).

| Data | Property IDs / source | Notes |
| --- | --- | --- |
| Class, subject | `001A`, `0037` | Only `IPM.Note` and subclasses count as emails; other IPM/REPORT classes, unsupported and unknown classes counted separately |
| Sender | `0C1A`, `0C1F`, `0C1E`, `5D01` | Name, raw address, address type, SMTP address; Exchange addresses are not resolved to SMTP |
| Recipients | `0E04`, `0E03`, `0E02`; sample recipient table `3001`, `3002`, `3003`, `39FE`, `0C15` | Display To/CC/BCC plus available name/address/type/SMTP/recipient-type rows; no address-book lookups |
| Dates | `0039`, `0E06`, `3007`, `3008` | Client submit, delivery, creation, modification; libpff naive FILETIME datetimes interpreted as UTC |
| Identifiers | `1035`, `3013`, `0070`, `0071`; libpff item identifier | Internet Message-ID, conversation ID/topic/index, local item ID; binary values hex-encoded privately; IDs are not assumed unique or portable |
| Replies/references | `1042`, `1039`, `007D` | MAPI In-Reply-To/InternetReferences and raw transport headers; separately parsed Message-ID/In-Reply-To/References values preserve provenance |
| Flags and other metadata | `0E07`, `0E17`, `0E08`, `0017`, `0036`, `3FDE`, `3FFD` | Message flags/status/size, importance, sensitivity and encoding hints |
| Bodies | `1000`, `1013`, `1009` | Presence-only inventory; sample plain text then HTML-to-text fallback; RTF-only is reported without text extraction |
| Attachments | libpff count; `3707`, `3704`, `0E20`, `370E`, `3705`, `3712` | Long/short filename, declared size, MIME type, method and content ID; metadata read only for samples; no `3701` data, buffer reads, embedded-message traversal or external references |
| Format | Header signature/version, libpff content/encryption/codepage | Numeric format evidence is reported; source is never repaired or rewritten |

Direction is cautious evidence: the unsent flag gives `draft`; the from-me flag gives `sent`;
otherwise a delivery timestamp gives `received_evidence`, with `unknown` for the remainder.
These are not verified mailbox-owner sent/received counts. A message's absence of from-me
alone does not prove receipt. Earliest/latest dates cover available submit/delivery times only.
Replies use reply/reference metadata or common Re/Rép prefixes; forwards use Fw/Fwd/Tr
prefixes. Localized prefixes and messages lacking these hints can be missed. No threading,
deduplication, owner identity resolution, signature removal or quoted-reply cleanup occurs.

Body availability counts mean properties exist, including possibly empty or malformed bodies;
inventory cannot distinguish empty body values without reading them. `plain`, `html_only`,
`rtf_only`, `none`, and `unknown` have this presence-only meaning. Actual sample reads report
empty/failed text independently, alongside property states. When plain text succeeds, HTML is
not fetched. HTML conversion ignores script/style/head content and preserves visible text;
it is not a browser render. Unicode strings are retained; byte bodies try BOM/UTF-8, then the
declared codepage, then explicit UTF-8 replacement. Decoding/truncation are recorded. No
attachment contents, RTF text extraction, arbitrary named-property mapping, or multivalue
property decoding is supported. Unknown property values are not fetched speculatively.

## Verification

Using the normal development environment:

```powershell
python -m pytest tests/test_pst_discovery.py -q
python -m flake8 src tools tests/test_pst_discovery.py --max-line-length=100
python -m black tools tests/test_pst_discovery.py --line-length 100 --check
python -m pytest --maxfail=1 --disable-warnings -q
```

Tests use synthetic parser objects, temporary PST-shaped byte files and fake metadata only.
They cover deep/recursive folders and cycles, missing/empty/error/unsupported fields, HTML
fallback, deterministic diverse sampling, limited inspection, attachment-data avoidance,
malformed items, interrupted/failure reports, private/structural separation, and read-only and
output-collision behavior. Native compatibility is a separate smoke check; no real user emails
are needed for automated tests. Database schemas, ingestion, web routes and production
dependency files are unchanged.
