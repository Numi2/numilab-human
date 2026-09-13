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
The fullbody vascular package admission is retained at [`Docs/FULLBODY_VASCULAR_ADMISSION_20260913.md`](FULLBODY_VASCULAR_ADMISSION_20260913.md) with its native receipt, physical-M4 CTest log (3/3), direct probe log, native identity, and SHA-256 manifest under [`Docs/media/organ-blood-cavity-bridge-20260913/native-fullbody-vascular-admission/`](media/organ-blood-cavity-bridge-20260913/native-fullbody-vascular-admission/). It proves source-independent package wiring on the real 157-body topology. The follow-up [`Docs/COUPLED_SUPPORT_REQUALIFICATION_20260913.md`](COUPLED_SUPPORT_REQUALIFICATION_20260913.md) records a successful synthetic exact-clock coupled root with ten valid support receptors at a stable fixture pressure and a retained fail-closed negative-pressure case. The execution registry therefore leaves force convergence over a calibrated range, anatomical supports/loading, activation, unresolved materials, calibration, standing, and walking as open gates.
The current ABI39 blood-owner refresh is retained at
[`Docs/BLOOD_MOMENTUM_RESTORE_20260913.md`](BLOOD_MOMENTUM_RESTORE_20260913.md)
with a physical-M4 build, 4/4 focused CTest selection, time-integrated
first-moment closure, synthetic pressure-gradient fluid momentum, and atomic
coupled-FEM/vascular restore plus fail-closed invalid restore. It closes only
the synthetic restore subgate; the source six-vessel bridge still lacks tube
geometry, lumen area, calibrated material/density, body-link mechanics,
tissue-side exchange, and subject calibration.

The physical M4 Pro also admits the exact BodyParts3D torso source surfaces
through the native MyoSim visual owner. The
[`native torso anatomy receipt`](NATIVE_TORSO_ANATOMY_20260913.md) renders five
organ, six vessel, and one spinal-cord surface across four hash-bound camera
views at the canonical `12.5 us` clock. It closes source membership,
source-to-world visual registration, and one-link kinematic visual binding;
the receipt deliberately leaves organ and vessel mechanics, materials,
density, blood/tissue exchange, calibration, loading, standing, recovery, and
walking unqualified.

The follow-up [`native stance horizon`](NATIVE_STAND_STANCE_HORIZON_20260913.md)
repeats the same physical M4 run from the authored support stance: six foot
witnesses carry the static `952.864475177 N` wrench with a
`1.86146132819e-6 N` root residual, and the compiled source activation is
balanced. The exact-clock 512-step replay remains dynamically divergent
(`108763.164 m/s²` peak acceleration, `1.30699` velocity delta,
`0.00444473` configuration delta), so it closes source stance placement and
static support/activation admission only; force convergence, sustained
standing, recovery, walking, calibration, and material/blood exchange remain
open.

The refreshed registry snapshot is retained at
[`Docs/media/gap-execution-20260913/report-with-native-stand-stance-20260913.json`](media/gap-execution-20260913/report-with-native-stand-stance-20260913.json)
with report SHA-256 `534393a5f2c555d46cbdf6499efef52852462220fec395e40c952baa11e35109`.
It remains `integrated_qualification: not_assessed` across 14 workstreams,
46 tasks, and 95 mandatory targets.

The current native persistent-stand owner has been exercised on the physical
M4 Pro at all four common durations (`100/50/25/12.5 us`) using the authored
six-contact stance, all 416 source routes at activation `1.0`, NHTENDON3
transfer, and no root assistance. The [exact-stand refinement receipt](EXACT_STAND_REFINEMENT_20260913.md)
records bitwise replay and a stable `1.87e-6 N` static root-force residual,
but `compiled_stand_balanced=false`, roughly `8.67e3 m/s^2` terminal
acceleration, and configuration drift over `0.8 ms`. This is retained partial
engineering evidence. It identifies the unresolved loaded dynamic
joint/fibre/contact state as the next owner and does not promote force
convergence, sustained standing, recovery, walking, or anatomical/calibrated
qualification.

The follow-up [native support-force horizon receipt](NATIVE_SUPPORT_FORCE_HORIZON_20260913.md)
removes the CLI's artificial 64-step ceiling and admits the declared 4096-step
solver bound. A physical M4 Pro run completes 512 exact `12.5 µs` steps
(`6.4 ms`) with static `952.864475176 N` support balance and bitwise replay,
but reaches `114347.445 m/s²` peak generalized acceleration, `1.4058` velocity
delta, and `0.0047894` configuration delta. The longer horizon is therefore
accepted transaction evidence and a retained temporal-convergence failure;
it does not promote standing, recovery, walking, anatomical loading,
activation calibration, blood/tissue exchange, or material qualification.

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
(file SHA-256 `2bba072af2d1417ce4b8a7f9c51e49a27188db97f5235b5737ac3d729090c82a`,
report content SHA-256 `e9c75fcb5b4e2afca5271fb5c231a2f5664eec335c8cc482a805b7e8dcfbe446`).
It remains 14 workstreams, 46 tasks and 95 mandatory targets with
`integrated_qualification: not_assessed`.
All snapshots have 14 workstreams, 46 tasks and 95 mandatory targets;
integrated_qualification remains not_assessed.

The current registry snapshot after the coupled-support requalification is
[Docs/media/gap-execution-20260913/report-with-native-coupled-support-20260913.json](media/gap-execution-20260913/report-with-native-coupled-support-20260913.json)
(file SHA-256 `e1f2c8a4da8a4a294cd1603975ef297dc834de2568ca866c32e46e906e0bfc3d`, report SHA-256 `fad05703b0431fc60804b47ef5a6d30e424cb5ff6f00d0096fb0a801faef8e14`). It retains 14 workstreams, 46 tasks and 95 mandatory targets; `integrated_qualification` remains `not_assessed` because the coupled result is synthetic engineering evidence and the anatomical, calibrated, and sustained behavior gates remain open.

The latest registry snapshot also includes the exact-clock persistent-stand
refinement receipt at
[`Docs/media/gap-execution-20260913/report-with-exact-stand-refinement-20260913.json`](media/gap-execution-20260913/report-with-exact-stand-refinement-20260913.json).
Its file SHA-256 is `d28df51c85c91d6bb3975808d86fd4c5683a652c2c9946ef8fad96edaaddf0f9`,
its report SHA-256 is `94e703d6cb9fc45bd097502fb35961efbdf6782b02420d584d9f2e045f09c261`,
and `integrated_qualification` remains `not_assessed` because the native
terminal still reports `compiled_stand_balanced=false` and dynamic drift.

The current source-data snapshot also includes the six-vessel source-to-world
registration receipt at
[`Docs/media/organ-vessel-registration-20260913/registration.json`](media/organ-vessel-registration-20260913/registration.json).
The receipt SHA-256 is `fb446cd380bac11aaa6ef8de2f011982da7a6aaaab981a4dcd2351583a69d9be`.
The corresponding registry snapshot is
[`Docs/media/gap-execution-20260913/report-with-vessel-registration-20260913.json`](media/gap-execution-20260913/report-with-vessel-registration-20260913.json)
with file SHA-256 `3719b46fb4b9d00aa2283e5c21d2c14f653b01e458208490f23aec7bc9d08c48`
and report SHA-256 `38ee2a43d6ce9b7d1036cd1f7d9ca4e40955b6d3e7143f47b7de12aa1740ed23`.
It closes source-to-MyoSim-world frame bookkeeping only; body-link vessel
mechanics, tubular fields, direction/area, density, blood ownership, tissue
exchange, calibration, and sustained behavior remain open.

The current ABI39 blood-momentum restore snapshot is
[`Docs/media/gap-execution-20260913/report-blood-momentum-restore-current-20260913.json`](media/gap-execution-20260913/report-blood-momentum-restore-current-20260913.json)
(file SHA-256 `00803c8e93b18737839e71b8ae835002dfd09729fdaeca849bca9310cb36b74b`,
report content SHA-256 `00774d5a95ba8a1d799be37793e7b4342cdee0199b175abddc10a8b15d4f89a5`).
It records 14 workstreams, 46 tasks and 95 mandatory targets with
`integrated_qualification: not_assessed`; the new atomic restore is a synthetic
blood-owner subgate and leaves source-bound vessel mechanics, calibration and
sustained behavior open.
