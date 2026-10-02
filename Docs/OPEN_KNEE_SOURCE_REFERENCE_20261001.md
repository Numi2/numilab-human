# Open Knee source-equivalent subsystem: reference admission

The shifted-ACL whole-body case is preserved as
`oks003-shifted-acl-reduced-hybrid-eight-microseconds` in
[`config/open-knee-shifted-acl-hybrid-regression.v1.json`](../config/open-knee-shifted-acl-hybrid-regression.v1.json).
Its source-bound receipt and binaries are unchanged. Eight accepted 1 us steps
are a contact regression, not evidence of source-equivalent knee mechanics.

## Implemented authoring boundary

`Source.mechanical_program` now retains the entire pinned FEBio XML program.
The new `open-knee-reference-case` command freezes the four original files,
compiles every XML element/attribute/text value in order, checks mechanical
references, and records the source rigid-joint graph. It requires no MyoSim
registration and makes no coordinate, constitutive, loading, or contact changes.
The program includes nested materials and prestrain, per-element meniscus
fibres, rigid-body definitions and ties, discrete springs, contact enforcement,
load curves, solver controls, and requested outputs.

The NHKNEE1 manifest now also emits a typed `source_mechanical_program`
inventory. Its checked record binds the 8,666,471-byte deck, archived FEBio
2.9.1 log, and DOI `Geometry_custom.feb` identity, all in the original source
coordinates. It lists the 21 material records, nine rigid bodies, 18 rigid ties,
six cylindrical joints, two other constraints, two prescribed-body boundaries,
18 sliding contacts, nine load curves, discrete interactions, and complete step
control values. Source material XML and every top-level FEBio section are hashed
and retained in the inventory. The current runtime admission remains explicitly
`rejected_unsupported_source_mechanics`; preserving this description does not
execute it. The cross-file contact-pair and rigid-tie references are explicitly
left unchecked against the archived geometry rather than resolved using the
different registered `Geometry.feb`.

The source inventory [receipt](media/open-knee-source-program-20261002/receipt.json)
and full [machine-readable record](media/open-knee-source-program-20261002/source-mechanics.json)
are retained.
The direct inventory compiler and its focused tests passed. Rebuilding the
registered NHKNEE1 payload manifest was not possible in this worktree because
the pinned `myo_sim` Python module is unavailable; no new registered payload is
claimed.

This is a source-program compiler and dependency audit. Matter lowering is
explicitly unsupported, all physical qualification flags are false, and the
original FEBio 2.9.1 solver has **not been reproduced**. Public comparison builds have now executed and their failures are retained separately. The recovered original run remains archived evidence. The command cannot promote its output to a
native mechanics result. The existing NHKNEE1 payload remains the reduced
hybrid and its generated manifest now states that boundary.

```sh
PYTHONPATH=src .venv-mujoco312/bin/python -m numilab_human.cli \
  open-knee-reference-case \
  --open-knee Sources/open-knee-oks003 \
  --matter-root /Users/home/MetalRobo-human-coupled-current \
  --output Build/open-knee-reference-20261001/retained-source-audit
```

The output directory must be new. The four-file retained bundle still exits **2** with six unresolved references. Supplying the recovered DOI archive closes those references and exits **0**, meaning ready for a reference run, not native mechanical qualification. A failed
admission still preserves the diagnostic artifacts. The original material
reference coordinates are retained byte-for-byte; no initialization solve or
energy reset occurs.

## New source-integrity findings

The original deck and original retained geometry do not form a complete
reference problem:

| Required construct | Retained bundle finding |
| --- | --- |
| `Geometry_custom.feb` | Missing; the deck names an absolute Windows path to this file |
| `QAT_@_QSO_TiesNodes` | Missing from retained `Geometry.feb` |
| `MPFL` discrete set | Missing |
| `LPFL` discrete set | Missing |
| `MCL_MNS-M_tie` discrete set | Missing |
| `FMC_To_QAT` contact pair | Missing; `QAT_To_FMC` is a different, unused definition |

The deck executes 18 sliding-elastic contact entries; the retained geometry
names 19 surface pairs. `MCL_To_MNS-M` and `QAT_To_FMC` are unused by those
contact entries. These counts must not be conflated.

Every active contact explicitly has `laugon=0`, `two_pass=1`,
`auto_penalty=1`, and `penalty=0.1`. Node relocation is not explicitly set in
this deck; its behavior cannot be inferred without the pinned solver's default.
The authored step is static: 40 increments of 0.05, adaptive bounds
0.001–0.05. Prestrain curves ramp ACL to 1.016, MCL to 1.034, LCL to 1.027
between source times 0 and 1. QAT, PCL and PTL curves remain at 1.0. The
flexion load curve stays zero to time 1 then reaches one at time 2, scaling
`Extension_Flexion` rotation to **-1.57 radians**. Those are static continuation
parameters, not evidence of two seconds of physical dynamics. There is no
explicit quadriceps force-loading section in this passive-flexion deck.

The patellar graph is `FMB(4) → PFFO(20) → PFPO(19) → PTB(1)`.
The neutral spatial motion map has rank six. This is a calculation from source
axes/origins, not validation of flexed body transforms, joint reactions, or a
native knee solve. All original intermediate bodies remain represented.

## Recovered matching reference

After explicit license acceptance, the [DOI archive](https://doi.org/10.18735/b0zv-n395)
provided the matching customized geometry, original log, deck, metadata, and
five processed CSV files. The archived deck and metadata are byte-identical to
the pinned inputs. The 119,032,115-byte customized geometry has SHA-256
`4155db1d0d7b87ffb2c668102d2495870e4461a539b18e6708f1f4817b5601bf`.
All mechanical references now resolve; the only unused surface pair is
`MCL_To_MNS-M`. This does not modify the original four-file hybrid source bundle.

Hashes and provenance are pinned in
[`config/open-knee-oks003-reference-archive.v1.json`](../config/open-knee-oks003-reference-archive.v1.json).
The archive's MIT notice is retained separately from the original bundle's
CC BY 4.0 notice. The different SVN revision-3413 files remain comparison
artifacts and are not substituted.

The recovered log identifies **FEBio 2.9.1**, normal termination, 140 accepted
increments ending at continuation time 2, and an original elapsed solve time
of approximately 7 h 54 min. It also contains **185 negative-Jacobian trial
diagnostics** and 19 warning blocks. The successful footer does not erase
those trials or establish strict nonintersection. Continuation time belongs
to a quasi-static loading program; it is not physical dynamic duration.

The log reader retains all **840** observation records: rigid center of mass,
quaternion, force and torque reactions, and connector forces and moments.
It checks body/connector IDs, record completeness, monotonic accepted times,
and printed time precision. Missing records, nonfinite values, version mismatch,
nonzero process exit, and failed termination cannot become a successful run.
Archived output and a fresh local reproduction have separate receipt fields.

```sh
PYTHONPATH=src .venv-mujoco312/bin/python -m numilab_human.cli \
  open-knee-reference-case \
  --open-knee Sources/open-knee-oks003 \
  --archive Build/open-knee-reference-20261001/doi-archive \
  --matter-root /Users/home/numi-open-knee-source-20261001 \
  --output Build/open-knee-reference-20261001/new-reference-case
```

Add `--febio /absolute/path/to/febio2` to execute the frozen problem. The runner
hashes the binary and inputs, relocates only the authored geometry include,
retains stdout/logs/observations, and refuses to overwrite an earlier attempt.
It only reports completion of the source protocol for FEBio 2.9.1, successful
exit, complete observations, and final continuation time 2. Original executable
build identity is still unknown; a matching version alone does not prove an
identical original binary or numerical backend.

No FEBio account is required for the new public-source comparison route.
[`tools/open_knee_reference`](../tools/open_knee_reference/README.md) builds pinned
public FEBio 2.9.0 and 3.0.0 sources with SuperLU/Apple Accelerate. Neither is
silently relabelled as the original 2.9.1 executable. Source mechanics files are
unchanged; portability changes affect optional linear backends and their failure
handling. Build identities and all three attempted full-deck runs are retained in
[`media/open-knee-reference-public-20261001`](media/open-knee-reference-public-20261001).

The 2.9.0 build rejects the prestrain material. The unadapted 3.0 build rejects the
legacy outer fibre field. An explicit 3.0-only frame translation then loads the
complete mesh and assembles **495,960 equations / 19,903,986 stiffness entries**,
matching the archived counts. It preserves the source axes through parent
material frames and local elastic fibres. During its first SuperLU factorization,
RSS reached **18,891,248 KiB**, crossing an explicit 18 GiB limit on the 24 GiB
Mac mini. The guarded process was terminated with **zero accepted increments**.
These remain unsuccessful baselines; the login barrier is removed but reference
reproduction is not complete. The raw failed receipt predates support for the
3.0 version-banner format; an additional reparse identifies 3.0.0 without changing
the retained original receipt.

## Archived contact and native material checks

The complete original XPLT is retained locally: **5,345,476,723 bytes**, SHA-256
`c370ae9f94e9faee2d7060bf2a6819e03be1312e82ca79bd2e9cebf8b34398de`.
All **141 plot states** (initial state plus 140 accepted increments) align with
the original text-log observations within float32 time rounding (maximum
6.28e-8). Contact fields cover all 36 authored contact surfaces. The plot also
retains displacement, reaction forces, stress, prestrain stretch and fibre
stretch fields for future spatial comparisons.

[Recovered source receipt](media/open-knee-reference-20261001/recovered-source-audit.json),
[contact checkpoint summary](media/open-knee-reference-20261001/archived-contact-summary.json),
[complete contact history](media/open-knee-reference-20261001/archived-contact.json.gz),
and [rigid observations](media/open-knee-reference-20261001/archived-observations.json.gz)
are retained with the archive's MIT notice. Exact root/state byte ranges at
indices 20, 60, 100 and 140 were also fetched independently and verified against
the completed file. The subset is labelled as an extraction, never as the full
original archive. Those checkpoints are near the proposed neutral/30/60/90
conditions; their archived processed tibiofemoral rotations are -0.00874,
-29.5350, -59.3848 and -89.6766 degrees. They are not exact prescribed-angle
qualification cases.

`open-knee-reference-contact --plot PATH --output NEW_JSON` reads the original
uncompressed XPLT version 5 contact gap and pressure fields, preserving surface
IDs, face counts, array hashes, and source units. It rejects unsupported
compression/layouts and incomplete fields. Partial downloads expose only complete
states and return status 2; they do not establish full-archive completion.

At the first accepted preload increment (continuation 0.05), the original
patellofemoral output has positive pressure on **7 of 22,478 femoral faces**
and **15 of 11,053 patellar faces**. Their respective maximum pressures are
0.0911842 and 0.182069 MPa. These are archived FEBio face values, not a native
contact result or a clinical interpretation. They provide a localized contact
baseline against which the broad foundation response can be compared.

The isolated Matter worktree at `/Users/home/numi-open-knee-source-20261001`
starts at the pinned `fac41f55` revision. It exposes compiled stored-energy
bytecode, removes default numerical viscosity from the source material, and
checks full energy/stress/tangent for all six tendon/ligament parameter sets,
both menisci, and the shared cartilage law. The independent tensor oracle
covers oblique fibres, transverse isochoric prestrain, shear and volume change.
Actual Apple M4 Metal execution checks 13,824 scalar values. The largest
stress/tangent Frobenius relative error is 7.0947e-6 against a 1e-5 limit.
Component error near cancellation is retained separately (maximum scaled error
9.17732e-4); the original scalar-case gates remain unchanged.

The new compiled public FEBio 3.0 material comparison also passes **486**
cases, including the actual prestrain wrapper and isochoric generator. Maximum
Frobenius errors are **2.954e-15** for first Piola stress and **9.254e-15** for
directional tangent (normalization floor 1 MPa). The source tangent independently
matches stress finite differences within **7.856e-9**. These are equation-level
comparisons, not assembled tissue equilibrium or a live tendon force-path repair.

Energy needs a separate qualification: the public build has no GSL, so legacy
fibre energy reporting is unavailable; the source's GSL high-stretch expression
also lacks the constant matching the toe energy at the transition. The 3.0
prestrain wrapper does not override deviatoric energy reporting. Matter's
continuous stored energy remains verified against the force law, but **legacy
energy-output equivalence is not claimed**. Exposing energy bytecode still does
not supply assembled runtime energy accounting. The source material defaults to
zero dissipation; the preserved hybrid explicitly requests 25 Pa s.

Matter now contains a native CPU/Metal cylindrical-connector residual and exact
directional derivative, preserving finite penalties, both free/prescribed
coordinates, attachment lever arms and explicit multipliers. Against the compiled
public 2.9.0 connector at **840** identical source-pose inputs, maximum force
error is **1.452e-10 N**, and connector-moment error is **1.772e-4 N mm**. The
latter includes the legacy quaternion `acos` precision loss near zero rotation.
The operator also passes 2,016 finite-difference component checks, mm/metre unit
conversion, and 1,035 Metal rows (maximum scaled CPU/Metal error 2.580e-7).
It is **not yet connected to the assembled knee solve**. Archived poses have
rounded precision and are not used as exact reaction truth. No removal of three
MyoSim equality rows is substituted for the source graph.

## Required continuation

1. Complete an independently converged reference comparison in specimen coordinates.
   The pinned public builds remove login dependence; the first full comparison
   remains memory-limited and unsuccessful. Preserve its baseline.
2. Lower the complete rigid/joint graph and exact material/prestrain laws into
   existing Matter, with source solver energy/stress/tangent comparisons.
   Continuum fibre restoration must retire the corresponding reduced force.
3. Implement all articular volume mechanics and joint/contact initialization,
   retaining material reference, prepared current coordinates, and prestrain
   separately. Qualify source contact and strict barrier admissibility separately.
4. Add an independently sourced quadriceps-loading experiment at the proximal
   QAT boundary, with unforced PTL tension, full wrench/virtual-work accounting,
   and coupled equilibrium. Passive flexion alone cannot qualify this experiment.
5. Complete swept-contact, refinement, energy/momentum, rollback, and sustained
   loading gates before replacing the local MyoSim prescription in the whole body.

The integrated native force path and contact kernel remain unchanged. The compiler,
material and joint-operator changes above do not qualify the assembled subsystem. Existing Matter worktree changes belong to other work
and were preserved. The remaining work above is not complete.

## Verification

Forty-two focused Human tests cover missing attachments/springs/contact pairs,
load-curve and rigid-body references, complete distributed fibre indexing,
unknown-field retention, cylindrical-chain rank, source hash rejection,
byte-preserving freezing, and refusal to overwrite an earlier case. The
existing importer, cartilage-material, and extensor-stack checks are included. Four current native tests cover source material and cylindrical-joint CPU/Metal execution; prior material-frame and stateful-package checks are retained. The live legacy importer still
reads 16 regions / 248,236 nodes and now retains all nine FEBio sections.

## Publication and continuation location

Human's reference authoring and readers are scoped to its `main` branch.
Native commits `081072b5` and `7846d6ff` are published on
`codex/open-knee-reference-20261001`, with
[draft PR #6](https://github.com/Numi2/numi-lab/pull/6) targeting the owning
`coupled` branch. Their isolated checkout is
`/Users/home/numi-open-knee-source-20261001`, based on `coupled@fac41f55`.
The prescribed production branch `numisolver` (inspected remote head `00feb4ae`)
does not contain the Matter tree. The draft does not migrate unrelated branch
history or claim production qualification. The shared dirty Matter checkout
and its unrelated changes remain intact.

The frozen case with complete archived evidence is
`Build/open-knee-reference-20261001/reference-with-complete-archive`.
The original outputs live in `Build/open-knee-reference-20261001/doi-archive`.
No downloader or simulation remains running for this task. No local reference
solver execution, native source-joint/contact solve, quadriceps transmission,
whole-body replacement, or numerical-convergence qualification is claimed.
