# One adult male: measured whole-body completion

The active release target is one source-bound adult male at one sourced age.
Age/population generalization is outside this release. Existing atlas, MyoSim,
cardiac and specimen sources remain independent sources; they are not measurements
of one person. No complete single-participant bundle has yet been admitted.

The historical 420-trial evaluator remains available as regression tooling. Its
survival and speed criteria do not establish whole-body completion.

## Executable authoring

```sh
numi human capability-protocol --requirements
numi human capability-protocol --protocol /path/to/protocol.json \
  --output /path/to/new-compiled-protocol.json
```

Use `.numi/commands/human` directly when the installed suite dispatcher resolves
a different workspace. The CLI is also `PYTHONPATH=src python3 -m
numilab_human.cli capability-protocol`.

The compiler requires 25 scenarios across balance/recovery, locomotion, terrain,
body transitions, manipulation, combined tasks and sustained activity. Required
scenarios are emitted by `--requirements`; they are coverage cells, not a fixed
number of trials. Each validation trial must belong to its cell and carry its
own required measured modalities. A second trial cannot supply missing forces
for an unrelated motion trace.

The [authoring schema](../schemas/human-capability-protocol.v1.schema.json) is
`numi.human.capability-protocol.v1`. Its `$defs` also describe the supporting
records. Structural schema validation does not replace compilation. It contains:

- `subject`: namespace, participant ID, male sex, sourced age >=18, and a hashed
  demographic record. The record includes the exact raw source and row locator.
- `model`: HumanPack bytes and a subject/age/parameter binding record. This is
  authoring provenance, not native proof of geometric or mass registration.
- `trials`: IDs, calibration/validation roles, session IDs and hashed measured
  trial records. Both roles are required; sessions must be disjoint.
- `cells`: all required scenario IDs, held-out trial IDs and comparisons binding
  measurement IDs, units, metric, positive tolerance, numerical error limit and
  a matching empirical-margin source record.
- `freeze`: a statistical-plan artifact, declared pre-selection freeze, 95%
  confidence and a numerical-error fraction no larger than 0.1.

Every artifact reference is `{path, sha256, bytes}`. All nested paths resolve
relative to the top-level protocol directory. No implicit download or search
occurs during compilation. Every artifact is checked again before publication;
changed output cannot overwrite an existing compiled protocol.

Supporting record schemas are `numi.human.subject-demographics.v1`,
`numi.human.subject-model-binding.v1`, `numi.human.measured-trial.v1` and
`numi.human.empirical-margin.v1`. Fields are strict. Measured trial records bind
the same namespace, participant, age, trial/session identity, capability ID,
conditions and measurement list. Measurement records distinguish `measured`,
`derived` and `simulated`; only measured records satisfy required empirical
comparisons. Raw measurement bytes cannot be reused as independent trials or
across calibration/validation splits. Derived data can remain in the inventory
but cannot be silently promoted to direct evidence.

Margins must bind units, metric and tolerance and cite measurement repeatability
or a source-validated margin. Supported comparison declarations are `rmse` and
`maximum_absolute_error`. This increment checks their declarations; it does not
calculate those errors, statistical confidence or physiological outcomes.

The compiled result always reports empirical/runtime qualification and source
authenticity as `not_assessed`, and freeze chronology as `not_attested`. Hashes
prove byte identity, not truth, independence of the original experiment, a
chronological preregistration, or successful native execution. No synthetic test
fixture is a selected participant or a completed capability.

## Execution dependencies

1. Admit a real participant with source demographics and an explicit anatomy,
   parameter and empirical coverage inventory. Keep unavailable cells open.
2. Migrate exact clocks through Brain, NumanX, sensors, circulation and restore.
   The current integer-microsecond transaction rejects 12,500 ns; changing only
   NHINIT admission would incorrectly round downstream timestamps.
3. Close full source-force consistency and 100/50/25/12.5 microsecond refinement.
4. Qualify loaded anatomy, supports/contact, prestress, reference reconstruction,
   activation, blood mass/momentum and calibrated materials with single ownership.
5. Lower the complete native task/observable program, prove accepted-only data
   production, and evaluate held-out responses with the frozen statistical plan.
6. Qualify sustained source-muscle control and task compositions on the same
   immutable subject/runtime stack. Profile native execution before long runs.

Different subjects, anatomical components or specimens can support separately
named component tests. They cannot be relabeled as this participant. Population
tasks remain in the broad roadmap; the selected single-male release does not
claim to close them.

## Verification, 2026-09-13

On the physical Mac mini, Python 3.13.14 passed 14 capability-contract tests,
12 gap-registry tests and 44 historical behavior tests in an isolated temporary
directory. Local Python 3.14.6 passed the same groups. Draft 2020-12 schema
checking and validation of the protocol plus 90 supporting synthetic records
passed with `jsonschema` 4.25.1 in an isolated local environment; the default
local Python initially lacked that optional package.

The [receipt](media/single-male-protocol-20260913/receipt.json) binds source and
archive hashes to the [Mac mini log](media/single-male-protocol-20260913/macmini-tests.log).
The log's `status=invalid` is an expected historical negative-control output;
all 70 tests passed. These tests do not run Metal, migrate clocks, admit real
measurements or close any anatomical, physiological or sustained behavior gate.
