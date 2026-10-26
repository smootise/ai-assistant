# Email signature research and benchmark

Research snapshot: **26 October 2026**. Base: `0b7b2f9`, preparation rules
`email-prepare-v1`. This is an isolated experiment under `tools/`, with no changes to
production detectors, schemas, original/processed emails, prepared chunks, or service clients.
No real mom database was opened during development. Stop at synthetic validation until review.

The operator's initial 25-email trial reported 61,775 normalized characters, 91 excluded
signature characters, 257 chunks, 232 unique chunks and no failures. These are supplied
observations, not ground truth or results reproduced here. Low exclusion alone does not prove
poor recall; the recognizable retained signatures justify a labeled evaluation.

## Candidate research

Versions and maintenance below were checked against upstream source and PyPI metadata,
not inferred from download counts. Installation/API compatibility was then tested on Windows
with CPython 3.14.3. License descriptions are upstream declarations, not a dependency license
audit. All three support extracted plain text and local execution for the selected APIs.

| Candidate | Actual API used | Languages, replies and nested signatures | Installation and maintenance |
| --- | --- | --- | --- |
| MAIF Melusine 3.4.0 | `Segmenter().segment_text(body)` then `ContentTagger().tag_email(messages)`; select `message.tags` with `base_tag == "SIGNATURE"`, using `base_text` | French-oriented processor with English rules; segmenter recognizes French/English Outlook fields, transfer markers and reply headers. Tags each segmented message. Wrapped/unrecognized headers and interleaved quoting remain imperfect. `GREETINGS` and `FOOTER` are separate labels; this adapter does not automatically discard everything after a greeting. | Python >=3.10; Apache-2.0. PyPI release 2026-09-01; upstream head `7c130d7507a198fcdbc809ed8c98c4ae6a27cdeb` dated 2026-09-01, not archived. Imports pandas, Arrow, OmegaConf and other framework components; distribution also requires scikit-learn/tqdm. Windows wheels installed successfully here. |
| Mail Reply Parser 1.36 (`mail-parser-reply`) | `EmailMessage(text=body, languages=["fr", "en"], keep_hyphen_lists=True).read().replies`; each reply exposes `.signatures` separately from `.disclaimers`, `.headers`, `.body` | Explicit French/English dictionaries; splits multilingual reply/Outlook headers, then finds signatures inside each reply. Repeated and deeply quoted signatures can be ambiguous. Signoff rules can consume the remaining reply, including a postscript or business instruction. Do not use `parse_reply()` to remove quoted history. | MIT; pure Python, no declared runtime dependencies or pretrained model. PyPI 1.36 and upstream `fc44bc880932f81f70a7deaf91c4525b3a6feddb` dated 2025-12-01; not archived. Package has no useful Python minimum declaration; import/API worked on 3.14.3. Smaller dependency and maintenance burden. |
| Mailgun Talon, source version 1.7.1 | Brute: `talon.signature.bruteforce.extract_signature(body)` returns `(body, signature)`. ML: `talon.init()` then `talon.signature.extract(body, sender=address)` returns the same pair. | Brute rules mostly target English closings/separators and some mobile strings, not general French business signatures. ML features use sender/contact evidence, with historical personal/Enron training; no validated French accuracy claim. Signature calls work on one body/tail; quotation APIs are separate. This benchmark does **not** strip quotations first, since quoted business facts must survive. Native nested-signature coverage is consequently limited. | Apache-2.0. **PyPI latest is old 1.4.4 (2017-08-24)**; repository is active, head `44d1fa5e5b511734f9ed2eeb727dc57218446c22` dated 2026-10-15. Pin that source revision, not an imagined PyPI 1.7.1 release. Modern source declares Python >=3.7, lxml/regex/cssselect/html5lib; `[ml]` adds numpy/scipy/scikit-learn/joblib. Both APIs worked on Windows 3.14.3 here. |

Primary references:

- Melusine: [repository](https://github.com/MAIF/melusine),
  [pinned processors](https://github.com/MAIF/melusine/blob/7c130d7507a198fcdbc809ed8c98c4ae6a27cdeb/melusine/processors.py),
  [package metadata](https://pypi.org/pypi/melusine/3.4.0/json).
- Mail Reply Parser: [repository](https://github.com/alfonsrv/mail-parser-reply),
  [pinned parser](https://github.com/alfonsrv/mail-parser-reply/blob/fc44bc880932f81f70a7deaf91c4525b3a6feddb/mailparser_reply/parser.py),
  [package metadata](https://pypi.org/pypi/mail-parser-reply/1.36/json).
- Talon: [pinned README](https://github.com/mailgun/talon/blob/44d1fa5e5b511734f9ed2eeb727dc57218446c22/README.rst),
  [setup/dependencies](https://github.com/mailgun/talon/blob/44d1fa5e5b511734f9ed2eeb727dc57218446c22/setup.py),
  [classifier loader](https://github.com/mailgun/talon/blob/44d1fa5e5b511734f9ed2eeb727dc57218446c22/talon/signature/learning/classifier.py),
  [PyPI metadata](https://pypi.org/pypi/talon/json).

Melusine's selected deterministic processors and Mail Reply Parser need no learned weights.
Talon brute needs none; ML loads a bundled serialized classifier. On this tested revision it
loaded successfully with scikit-learn 1.9.1, but model compatibility remains version-sensitive.
Joblib/pickle loading executes trusted Python serialization: only use the pinned, reviewed
distribution, never email-supplied or downloaded replacement model paths. No training on private
messages occurs. Optional Melusine transformers/connectors are outside this experiment.

The chosen APIs make no external inference/API calls. Installation downloads public software;
benchmarking uses no PST, LLMs, embeddings or Qdrant. Python socket connection/DNS entrypoints
are blocked during detector initialization/execution; third-party prints and logging are
suppressed. This guard is not an OS sandbox against arbitrary malicious native code. For private
evaluation, use a disconnected session or outbound firewall restriction as well. Keep generated
private files in an unsynced local folder. No vulnerability/CVE audit was performed. Native wheels,
regex worst-case behavior, dependency supply chains and model deserialization remain review areas.

No additional candidate is included: other reply-only parsers would repeat the quote-retention
risk without a demonstrated French signature advantage. This is a bounded comparison of these
implementations, not a claim that all available Melusine pipelines or Talon models were exhausted.

## Repetition experiment and combinations

`corpus` learns from a **separate, unlabeled reference set**. It requires three distinct body
hashes from the same normalized sender address, an explicit French/English signoff, matching
sender email contact and a telephone in a short trailing block. It normalizes case, whitespace
and phone separators for lookup, without changing source text. A business-keyword guard rejects
orders, prices, invoices, instructions and related facts. Repetition alone never qualifies.
Duplicate copies of one email cannot create support. Matching email bodies in reference and
evaluation sets are rejected rather than counted twice.

`hybrid` intersects Mail Reply Parser spans with these supported trailing blocks.
`baseline-hybrid` takes the union of that agreement and V1 baseline signature spans. This is
prototype evidence for a hybrid, not production authorization or calibrated confidence.

The prototype intentionally abstains on recognized quote/forward histories: attributing nested
signatures to the outer sender would contaminate the reference index. Library/baseline detectors
still evaluate those fixtures. A future corpus approach could use explicitly parsed per-message
headers to associate inner signatures with their actual sender; unknown/ambiguous attribution
must abstain. Neither body-wide repetition nor message frequency authorizes discarding repeated
order forms, product specs, newsletter offers, schedules or standard instructions. A phrase list
cannot eliminate all such false positives, even with contact evidence.

## Reproduce synthetic results

From the repository root, use a separate environment. No production dependency is added.

```powershell
.venv\Scripts\python.exe -m venv .venv-signature
.venv-signature\Scripts\python.exe -m pip install -r tools/requirements-signature.txt
$env:PYTHONPATH = "src;."
.venv-signature\Scripts\python.exe -m tools.email_signature_benchmark run `
  --input tests/fixtures/signature_benchmark.json `
  --reference tests/fixtures/signature_reference.json `
  --detectors baseline mail-reply melusine talon-brute talon-ml corpus hybrid baseline-hybrid `
  --report signature-results-new.json
```

Output files must be fresh. Exit 0 means every requested adapter ran successfully, **not** that
its accuracy is sufficient. Missing/incompatible dependencies produce `unavailable`; per-message
errors produce `partial`. Exit 1 covers either and invalid inputs; failures never count as correct
empty predictions. Aggregate output contains no body, ID, subject, sender, path or exception text.
Only metrics, fixed statuses, dataset hashes and software identity are recorded.

`tools/requirements-signature-windows-py314.txt` records the exact tested dependency snapshot for
Windows 3.14; the shorter requirements file is appropriate for CI Python 3.11 resolution.
The source-authored annotations live in `tools/signature_synthetic.py`; regenerate the two
committed JSON fixtures with `python -m tools.signature_synthetic`. Labels are independent of
all detectors. `source=synthetic-authored` explicitly distinguishes them from real operator
`source=manual-reviewed` annotations. They are **not** independent human validation of real mail.

Fixtures cover French/English closings, roles, company/location/contact/social lines, mobile
footers, separators, unmarked signatures, CRLF/indentation/Unicode, extracted HTML text, long and
empty bodies, Outlook/Gmail/forward/nested replies, identical inner/outer signatures, postscripts,
contact instructions, lists, order forms and newsletter offers. Four distinct synthetic reference
messages provide repeated-sender support. This small, deliberately adversarial set has selection
bias and cannot estimate production precision or the French archive's language distribution.

## Metrics and initial evidence

Full local results: [signature-synthetic-results.json](signature-synthetic-results.json).
Thirty author-labeled synthetic bodies, 22 signature blocks, no real mail; all eight adapters
completed without failed messages on Windows/Python 3.14.3.

| Detector | Character precision | Character recall | Critical false-positive characters |
| --- | ---: | ---: | ---: |
| V1 baseline | 100.0% | 19.6% | 0 |
| Mail Reply Parser | 87.3% | 81.8% | 187 |
| Melusine ContentTagger | 85.3% | 25.2% | 77 |
| Talon brute | 83.1% | 24.8% | 82 |
| Talon ML | 80.4% | 71.3% | 99 |
| Corpus prototype | 100.0% | 19.2% | 0 |
| Parser/corpus intersection | 100.0% | 19.2% | 0 |
| Baseline plus parser/corpus agreement | 100.0% | 38.8% | 0 |

Repetition complements the baseline in this constructed set, raising recall from 19.6% to 38.8%
without a tested false positive. Repetition alone is much weaker than the parser's recall;
intersection does not improve on corpus alone here. Nothing establishes that agreement helps
on real data. Every standalone library has critical false positives in these fixtures. Melusine
classifies small contact parts, whereas the parser commonly returns whole closing tails, which
helps explain their very different block/character behavior.

For each **successfully evaluated, reviewed** record, predicted/gold overlap gives TP characters;
predicted outside gold gives FP; gold outside predicted gives FN. Character precision is
TP/(TP+FP), recall TP/(TP+FN), and correctly identified signature percent is 100 times recall.
Exact-block precision/recall additionally require identical boundaries. `completely_missed_blocks`
counts gold blocks with zero overlap; `missed_signature_chars` captures partial misses too.
Critical spans separately measure deletion of business facts. Whitespace, closing phrases and
quote prefixes count as labeled; this strict policy can penalize boundary-only differences.
Metrics with zero denominators are null, not perfect scores. Failures and reviewed-message
coverage remain visible, so a failing adapter cannot win by abstaining from hard cases.

All records, including unreviewed ones, contribute only to `estimated_excluded_chars` and
`estimated_text_reduction = predicted_chars / input_chars`. These are automated source-character
estimates, not verified signatures, embedding token savings, final chunk sizes, deduplication
savings or measured costs. Production footer exclusions and formatting changes are outside the
signature-only baseline being compared.

Elapsed seconds include initialization/imports and corpus-index construction. Tracemalloc records
peak Python allocations; native buffers are excluded. `process_peak_rss_bytes` measures process
lifetime peak working set/RSS, including native allocations. Later detectors inherit earlier
imports/peaks; run each detector in a fresh process for independent resource comparisons. On the
recorded ordered run, process peak grew from about 36 MiB at baseline to 192 MiB after the larger
library imports. Small-fixture latency is not archive-scale throughput; tracemalloc also adds
overhead. Export currently scans/stores all eligible processed bodies and sections in RAM before
sampling, so memory scales with corpus characters. It does not claim streaming scalability.

## Review 50–100 real messages later: operator only

These instructions are prepared for **after code review**; they were not run during development.
No migration, preparation, PST read or reprocessing is needed. Stop writers and use a private
unsynced destination outside any Git repository. The output parent must already exist; all output
filenames must be new. Confirm the explicit DB path and profile yourself. There is no default
mom path or automatic profile/config lookup.

```powershell
$env:PYTHONPATH = "src;."
# Explicitly invoked read-only access; use the confirmed profile database path.
.venv-signature\Scripts\python.exe -m tools.email_signature_benchmark export-readonly `
  --db "D:\confirmed-profile-path\jarvis.db" --profile mom --count 75 --seed 20261026 `
  --allow-private-read `
  --private-output "D:\JARVIS_Private\signature-evaluation.json" `
  --private-reference "D:\JARVIS_Private\signature-reference.json"
```

SQLite opens with `mode=ro`, `query_only=ON`, `trusted_schema=OFF`, and one read transaction.
Missing databases, wrong profile markers or missing processing extensions fail without creation
or migration. The query reads successful `email-clean-v1` normalized bodies/sections from
imported messages in verified archives; failed latest processing attempts are excluded. Prepared
tables are never queried. Source connections close before artifact writing. Normal SQLite read-only
WAL access may require existing WAL/SHM availability; the tool never forces checkpoints or changes
journal mode. If the filesystem cannot support a read-only WAL connection, stop and use an
operator-created consistent backup; do not enable `immutable=1` on a live database.

Sampling shuffles within composite sender/length/reply-type/source-format strata using the seed,
then round-robins shuffled strata. It scans the population, not the first 100 rows. It improves
coverage but is not a probability-weighted production prevalence estimator; inspect coverage and
add manually selected edge cases separately. Missing sender metadata shares an unknown bucket;
reply types and original body source are heuristic metadata, not reconstructed threads. Fewer than
75 eligible rows produces a smaller sample, visible in the count. The optional reference file
contains the remaining eligible bodies, excluding **all exact copies** of evaluation bodies,
with sender/format metadata and no labels. It can contain the full remaining archive: keep it local.

1. Annotate the 75 evaluation records **before viewing detector outputs**. Preserve `body`,
   `sections` and `body_sha256`. Offsets are Python Unicode character indices, zero-based,
   half-open `[start, end)`, into exactly `body`, not PST bytes, UTF-16 indices, processed-text
   offsets or rendered lines. The exported body is the unchanged stored normalized representation;
   this experiment does not alter its normalization. Original source records remain in SQLite.
2. Set `label.signatures` to all verified signature spans, including quoted occurrences when
   appropriate. Mark business-critical instruction/price/product/contact-instruction spans in
   `label.critical`. Use `[]` for a reviewed negative. Ambiguous records remain `unreviewed`.
   Set `label.reviewer` to a local identifier, `label.source="manual-reviewed"`, then
   `label.status="reviewed"`. A matching body hash and valid non-overlapping ranges are required;
   critical and signature labels cannot overlap. Library predictions never become labels.
3. Have another reader check every positive exclusion and all business-critical cases; resolve
   disagreements locally. Record annotation policy/boundary changes and freeze a held-out set.
   Keep a separate development set for thresholds. Reference messages provide repetition evidence
   without evaluation labels; avoid reference/evaluation copies or reconstructed-thread leakage.
4. Run the benchmark on the labeled JSON, passing the separate reference file. Keep the report
   private until checking that only aggregate statistics are shared.

```powershell
.venv-signature\Scripts\python.exe -m tools.email_signature_benchmark run `
  --input "D:\JARVIS_Private\signature-evaluation.json" `
  --reference "D:\JARVIS_Private\signature-reference.json" `
  --detectors baseline mail-reply melusine talon-brute talon-ml corpus hybrid baseline-hybrid `
  --report "D:\JARVIS_Private\signature-metrics-new.json" `
  --private-review "D:\JARVIS_Private\signature-review-new.json"
```

The private review JSON preserves each complete input body, original metadata/sections/labels,
and each detector's spans/status/unmapped count. Open it in a local editor; compare slices
`body[start:end]` against gold, particularly every additional exclusion and each detector
disagreement. No content is printed by batch commands. Aggregate `disagreement_messages` counts
differences in span/status output, not accuracy or a vote establishing truth. Duplicate or
unmappable library strings are counted as `unmapped_candidates` and abstained; the harness never
uses fuzzy matching to invent source positions. CRLF and line-edge whitespace normalization use
an explicit character map back to the unchanged body. Private artifacts are rejected inside Git
repositories and never overwritten. They must not be uploaded as CI artifacts or committed.

With only 50–100 real messages, zero observed FP still cannot establish near-perfect precision.
Report the number of reviewed predictions, misses, critical losses and coverage by sender,
format, length and history type; require more independent evidence for rare cases. Do not tune
on this holdout and report the tuned score as validation. Stop after the reviewed evaluation;
production integration needs a separate decision.

## Preliminary recommendation and V2 proposal

Prefer a **conservative hybrid experiment**: preserve V1's recoverable decisions, add a lightweight
multilingual parser as a candidate proposer, and require independent repeated-sender/contact
evidence for additional removals. Uncertain boundaries, business evidence, unknown inner senders
or unmappable strings stay retained and reviewable. The implemented `baseline-hybrid` is a
minimal prototype of this policy. Its synthetic recall gain is promising; production superiority
and the value of library agreement still require independent real-data validation.

Mail Reply Parser's small dependency surface makes it the leading proposer to evaluate. Its
raw tail deletion is unsuitable at the requested precision priority. Melusine has useful French
segmentation/contact tags but brings a larger framework and insufficient full-block coverage in
this specific adapter. Talon's ML offers greater recall than brute but has historical model,
serialization/dependency and false-positive costs. Continuing a new standalone regex detector
is not justified by these results; limited business-evidence vetoes remain necessary safety
policy around an established extractor. A corpus index also costs memory, drift handling and
snapshot/version management. For rare/new senders it should abstain.

If approved later, introduce `email-prepare-v2` as a **new rules generation**, retaining base
v8/email/processing/preparation schemas unless a separately reviewed additive extension is needed.
Bind generation identity to the detector/configuration versions **and reference-corpus snapshot
hash**, not just chunk sizes. Store source processing fingerprint, exact spans, reasons, sender
attribution/support provenance, abstentions and model/package provenance. Corpus changes must
not silently change an existing generation's results. Avoid retaining private support text in
logs or global cross-profile caches; corpus evidence must remain profile-scoped.

Use existing backed-up, per-email transactional preparation storage to create parallel V2 chunk
contents/occurrences; keep V1 rows/IDs and all original/processed representations intact. A failed
V2 attempt preserves its prior successful derived rows; ordinary retries resume. Validate input
fingerprints and compare exclusions/chunk lineage before an operator explicitly selects V2 for
future indexing. Recompute V2 content hashes, offsets and exact sharing; do not rewrite old
occurrences, remove chunks, purge collections or regenerate embeddings as part of this research.
No schema migration, V2 integration or index switch is implemented here.

## Continuation checkpoint

Local validation on Windows/CPython 3.14.3: **1,016 passed, 4 skipped** in the full repository
suite; all 25 focused benchmark tests passed. The expanded CI Flake8 command and Black's
100-column check passed. All eight real candidate adapters/combinations completed synthetic
execution successfully. These checks used no real profile data.

Start by reading this document, the committed synthetic metrics, `tools/email_signature_benchmark.py`
and `tests/test_email_signature_benchmark.py`. Optional dependencies stay in `tools/requirements-signature*.txt`.
The ordinary full suite tests offsets, labels, failure/privacy handling, repetition guards,
sampling and synthetic read-only SQLite export. A separate CI job installs the pinned candidates
and exercises their real APIs against synthetic fixtures only. Never use the real profile to
debug a failing test. Review the commit before invoking the operator workflow. The next decision
is real-data labeling/evaluation authorization, not implementation of V2.
