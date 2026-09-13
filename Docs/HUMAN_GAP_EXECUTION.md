# Human gap execution registry

The completion plan is now an executable inventory. Its 14 workstreams match
the [live ledger](HUMAN_COMPLETION_GAP_LEDGER.md), its 46 tasks name owners and
acceptance criteria, and its target links cover all 95 mandatory Human targets.
The dependency graph has ten layers and twelve initial independent tasks.
These are scheduling relationships; tasks are not assessed for readiness or
completion by this command.

```sh
numi human gap-execution
numi human gap-execution --output /private/tmp/human-gap-execution.json
numi human gap-execution \
  --coverage /path/to/current-target-coverage.json \
  --output /private/tmp/human-gap-execution-with-coverage.json
```

The [registry](../config/human-gap-execution.v1.json) stores engineering,
source-data, calibration, validation and performance tasks. It contains no
editable qualification status. The command quotes current ledger status,
remaining gaps and closure gates, hashes every referenced file and emits a
content hash for the report. A changed report cannot overwrite an existing
output path. The [schema](../schemas/human-gap-execution.v1.schema.json) rejects
extra fields; the compiler additionally rejects missing ledger rows or
mandatory targets, duplicate identities, unknown prerequisites, cycles and
missing or external repository references.

With `--coverage`, the existing target-coverage validator first admits the
manifest. Every mandatory target is bound to its current leaf hash. Additional
source leaves remain explicitly unmapped, and unresolved source registers
remain visible. An absent coverage input yields `not_assessed` source assignment;
an empty source inventory does not imply complete source coverage. The command
does not validate the scientific evidence referenced by the ledger, lower a
runtime package, or change integrated qualification from `not_assessed`.

The first engineering streams are candidate/accepted-position precision
and accepted-root behavioral telemetry. The [bounded implementation](ACCEPTED_STATE_PRECISION_20260913.md)
repairs the retained 25 µs failure and adds a real joint-publication metric
producer. The [source-route follow-up](SOURCE_ROUTE_PRECISION_20260913.md) adds
paired muscle/hood geometry and full-model read-only source evaluation; both
tasks retain their full acceptance gates below. Anatomical
supports/loading, activation, spatial blood mass and momentum ownership,
unresolved material data and calibration have separate dependency tasks.
The registry retains the unchanged 100/50/25/12.5µs common-duration refinement
gate. The active behavior target is the [single-male whole-body protocol](SINGLE_MALE_COMPLETION.md);
the historical 420-trial standing/recovery/walking evaluator remains regression
tooling. Neither a shorter native trajectory nor metadata coverage closes these
gates. Population tasks describe the broader roadmap, outside this selected release.

The native behavior compiler and telemetry clock is now bound to the same
canonical 12,500 ns constant as the ABI8/v2 exact-clock runtime. The physical
M4 focused behavior and Human source selection passes 9/9; the retained
[receipt](HUMAN_BEHAVIOR_CLOCK_20260913.md) records `clock=12500ns` while
preserving `physical_steps=0` and unknown audit coverage. This removes a unit
inconsistency in the engineering path without changing the force-convergence,
anatomical, calibration, standing, or walking acceptance gates.

Validation on 2026-09-13: twelve focused registry tests and fourteen existing
target-coverage tests pass. The JSON Schema passes Draft 2020-12 schema checking
and independently validates the shipped registry with `jsonschema` 4.25.1.
These checks cover inventory integrity and report behavior, with no new
anatomical, mechanical or behavioral qualification.

The current no-coverage snapshot is retained at
[`Docs/media/gap-execution-20260913/report.json`](media/gap-execution-20260913/report.json)
(SHA-256 `e2763bb92599dae96d5bea68fa794e42f6076a3a940fb9b5647058babc9d919a`).
It records 14 workstreams, 46 tasks, 95 mandatory targets, and
`integrated_qualification: not_assessed`; it is a completion inventory rather
than a qualification receipt.

The prior source-to-hydraulic organ bridge and cardiac
material/activation/support/loading snapshot is retained at
[Docs/media/gap-execution-20260913/report-with-organ-blood-and-loading-native.json](media/gap-execution-20260913/report-with-organ-blood-and-loading-native.json)
(SHA-256 b19bdff7276001dc0f6f29f16bc1ac4e9a46541a1f9eff4d47d5f30b6e7f9cd7).
The ABI38 source-to-native identity/owner admission is included in the
intermediate no-coverage snapshot at
[Docs/media/gap-execution-20260913/report-with-native-organ-binding.json](media/gap-execution-20260913/report-with-native-organ-binding.json)
(SHA-256 44f3d8585601a89b1b8fcf40948f03006c9cf813855ef90f6c8482caaff90360).
The final post-ledger snapshot before tissue requalification is
[Docs/media/gap-execution-20260913/report-with-native-organ-binding-ledger.json](media/gap-execution-20260913/report-with-native-organ-binding-ledger.json)
(SHA-256 7a17975431d7911feefc9d8302c510f7e572633db8d0d3f68fdc100f2fda2372).
The current snapshot also includes the native costal tissue requalification:
[Docs/media/gap-execution-20260913/report-with-native-organ-binding-tissue-ledger.json](media/gap-execution-20260913/report-with-native-organ-binding-tissue-ledger.json)
(SHA-256 2a9fa4395f95ca9a485b13bbe305b8d05bb99a1272546177df694a6bfe965e79).
The latest snapshot also records the native ABI38 pressure-gradient wall
reaction path and its source-bound synthetic Jacobian receipt:
[Docs/media/gap-execution-20260913/report-with-native-organ-binding-pressure-reaction.json](media/gap-execution-20260913/report-with-native-organ-binding-pressure-reaction.json)
(report SHA-256 24ba96d13ebb908421985b796812014497ee1a1cf90a03a27740ba9abf291887).
The ABI39 native fluid-momentum owner receipt is recorded at
[Docs/FLUID_MOMENTUM_TRANSFER_20260913.md](FLUID_MOMENTUM_TRANSFER_20260913.md)
with its hash-bound native log and JSON receipt. The post-ledger report is
[Docs/media/gap-execution-20260913/report-with-native-fluid-momentum-v2.json](media/gap-execution-20260913/report-with-native-fluid-momentum-v2.json)
(report SHA-256 `eca0a63c6ba0648637e80c1c1a08afcd4b5107a9f337ecea7fa01e0ad17ddc40`)
and remains `integrated_qualification: not_assessed`.
The post-exact-clock snapshot is retained at
[Docs/media/gap-execution-20260913/report-with-native-exact-clock-20260913.json](media/gap-execution-20260913/report-with-native-exact-clock-20260913.json)
(report SHA-256 `ea2f7a2ed73cfdf538addde4e2ee2190e386dd817107edc35d1dd45ac9289715`).
It records the same 14 workstreams, 46 tasks and 95 mandatory targets; the
new native exact-clock receipt is reflected in the ledger while
`integrated_qualification` remains `not_assessed`.
The current native organ/blood and cavity requalification is paired with the
snapshot at
[Docs/media/gap-execution-20260913/report-with-native-exact-clock-organ-blood-20260913.json](media/gap-execution-20260913/report-with-native-exact-clock-organ-blood-20260913.json)
(file SHA-256 `4466b7cd475008ba1cbfd906bc07741bec000b570eba8702bf2a917881689f44`,
report content SHA-256 `4858362e8f42844dd334172d8fc20d9f6a8413c64d95a0d1b8e61be632a8f1b2`).
It still records 14 workstreams, 46 tasks and 95 mandatory targets with
`integrated_qualification: not_assessed`; the accompanying native receipt is
the focused 17/17 requalification at commit
`e07026ab3ad497a869ee247cdfd1aa23ecc16ebb`.
The next snapshot adds the aligned behavior clock receipt:
[Docs/media/gap-execution-20260913/report-with-native-exact-clock-organ-blood-behavior-clock-20260913.json](media/gap-execution-20260913/report-with-native-exact-clock-organ-blood-behavior-clock-20260913.json)
(file SHA-256 `f0a576df6b8cb5e601c19eab5f6a27df9f2507436424d98ef86454280e40b1ce`,
report content SHA-256 `89dbe4bd68f6a3be1c2a82d8927b9c0b5b2fc77089ed7135127dab45a353a2de`).
It remains 14 workstreams, 46 tasks and 95 mandatory targets with
`integrated_qualification: not_assessed`.
All snapshots have 14 workstreams, 46 tasks and 95 mandatory targets;
integrated_qualification remains not_assessed.
