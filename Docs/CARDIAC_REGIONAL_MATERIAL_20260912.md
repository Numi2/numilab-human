# Cardiac regional materials and source frames, 12 September 2026

The Human authoring layer now converts every Rodero case18 fibre/sheet pair into
an explicitly derived, source-bound material frame. An independent C++ reader
checks all 1,470,083 frames and their Float32 representation. The native regional
material extension retains shared FEM nodes while selecting each tetrahedron's
constitutive law and density. These steps do not yet assemble or simulate the
anatomical wall.

**PASS for bounded native regional materials and full-source frame preparation.**
Native commit [`3636bb3`](https://github.com/Numi2/numi-lab/commit/3636bb371ec78c0c1620323d98fcfdeff3fb8d3f)
is published on `coupled`. All **25 selected native checks** pass on the physical
Apple M4 Pro, with no failures or skips. The [qualification receipt](media/cardiac-regional-material-20260912/qualification.json)
binds the published source bytes, binaries, commands, host and retained attempts.
The anatomical source still has **zero physical steps**.

## Explicit source-frame conversion

The source is the attributed Rodero case18 CT-derived asset described in
[the anatomical source report](CARDIAC_WALL_ANATOMY_20260912.md) and pinned by
[cardiac-wall-rodero18.v1.json](../config/cardiac-wall-rodero18.v1.json), SHA256
`23c931fef53edced85e0e0a36c73d8490dddb87db6bd988482a2cdfc5a1442cc`.
The exact archive SHA256 is
`b50919a711dc3914cc0a4dd9cabc19679a9f36be9ac6eb26a23ca6799f501720`;
the imported asset manifest SHA256 is
`e8cb0391623577efc4eac04e5710cf7c9a4757614e09d936f5af3889c37d56f1`.
Its 300,965 nodes, 1,470,083 tetrahedra, raw axes and 24 regional labels remain
unchanged.

[`cardiac_material_frames.py`](../src/numilab_human/cardiac_material_frames.py)
requires the explicit policy `normalize-fibre-gram-schmidt-sheet-v1`. It
normalizes the fibre, removes its projection from the normalized sheet, and
normalizes the sheet and their cross product. The resulting rotation has
columns fibre, sheet and normal, mapping material axes to reference-world axes.
The output is a unit Hamilton quaternion `(x,y,z,w)`, with nonnegative `w` and
a deterministic sign tie-break at `w=0`. This changes the derived basis only;
the raw source axes are preserved. Source axes are constructed fields, not
measured microstructure.

The public `prepare_material_frames` API and
[`numi human-cardiac-material-frames`](../.numi/commands/human-cardiac-material-frames)
require the exact input manifest SHA and independently check all thirteen
pinned source buffers. Consumed fibre/sheet/label bytes are hashed during
conversion; all named inputs are rechecked before atomic publication. Invalid
axes, forged manifests, changed inputs, changed existing output and symlinks
are rejected. The identity binds raw inputs, declared conversion policy and
exact output bytes. The converter performs no physical stepping.

The isolated Mac mini command was:

```sh
PYTHONPATH=/Users/n/human-cardiac-material-frame-preparation-20260912/source/src \
/Users/n/human-cardiac-partition-20260912/venv/bin/python \
  -m numilab_human.cardiac_material_frames \
  --asset /Users/n/human-cardiac-wall-anatomy-20260912/asset-final \
  --output /Users/n/human-cardiac-material-frame-preparation-20260912/frames \
  --asset-manifest-sha256 e8cb0391623577efc4eac04e5710cf7c9a4757614e09d936f5af3889c37d56f1 \
  --conversion-policy normalize-fibre-gram-schmidt-sheet-v1
```

All 1,470,083 frames passed in **12.1846 seconds**, with **30,867,456 bytes**
peak child RSS. The 47,042,656-byte Float64LE buffer has SHA256
`8ab7c6f59e56852ecf4210042df766d73d6a18c1dca2b48d30e524601dbf0824`.
Its source identity is
`b4799ce922fc22a6730a521bafcb7f843c74ec8b62543c5529ae19b87f1decaa`.
All raw input and implementation hashes matched before and after execution.

| Full-source conversion diagnostic | Maximum |
| --- | ---: |
| Raw fibre / sheet norm deviation | `1.06509117e-5` / `1.01706982e-5` |
| Sheet direction correction | `1.48228549e-6 rad` |
| Derived orthonormality / quaternion norm-squared error | `4.44089210e-16` / `4.44089210e-16` |
| Reconstructed matrix component error | `8.32667268e-16` |

The compact [source-frame evidence](media/cardiac-regional-material-20260912/source-frames/REPORT.md)
retains the command, before/after source hashes, logs and diagnostics.
The generated manifest SHA256 is
`cc74e1ac10e74c602a57049a4a0ec36a9eab86cad88e4ef138becf59d64d5aeb`;
the execution record SHA256 is
`03912da1990bd2c7209fc356f4ed02cc1690ab6e92aa92449bdbac9d0c7ea3e3`.
The producer module SHA256 is
`b4689ca3144f095d9c5ba505145a9bc342511f5333d394d7fb9f42eb42f1aeb2`.

The 19 focused converter tests and 10 importer tests pass locally and on the
Mac mini. The first retained converter-test failure was an incorrect overflow
fixture expectation: the robust norm of `(1e308,1e308,1e308)` is finite. The
test now uses an actually overflowing vector; no conversion gate was relaxed.
Existing-output verification currently stages another full buffer. A streaming
optimization is deferred and does not change this qualified producer identity.

## Independent C++ frame check

[`cardiac_material_frame_check.cpp`](../tools/cardiac_material_frame_check.cpp),
SHA256 `8593e1365541bf5a91a1090d619bc48707f5e942cc575182338e559f34b875b7`,
independently reads the raw axes and output quaternions. It reconstructs every
source basis, checks canonical quaternion signs and norms, populates the native
`ObjectSource::femMaterialFrameRotations` and source-identity fields, then checks
the Float32 quaternion representation and normalization.

All frames passed in **0.52 seconds**, with **48,513,024 bytes** peak RSS.
The maximum source-basis error was `8.32667268e-16`; the Float32 quaternion
norm-squared error was `9.85279058e-8`, and reconstructed Float32 basis error was
`6.39173484e-7`, below the explicit `1e-6` gate. This checker does not cook a
whole wall world or execute a GPU step. Its result is source-field/native-API
compatibility evidence, not anatomical mechanics.

The retained [full-source result](media/cardiac-regional-material-20260912/cpp-evidence/full-source-attempt-001.json) has
SHA256 `2c0f66e474e60b751629bb3c73c38c0c209885b185d4c29d3e4e4ad7b783ddad`.
The [independent controls](media/cardiac-regional-material-20260912/cpp-evidence/controls/results.json)
accept three valid cases and reject ten invalid cases, including wrong basis,
nonunit/noncanonical quaternions, invalid axes, missing identity and shape
errors. Their exact commands and outputs are retained with the evidence.
The [C++ qualification record](media/cardiac-regional-material-20260912/cpp-evidence/qualification.json)
binds the build, binary and input hashes. The C++ reader decodes the supplied
source identity; this outer record authenticates the corresponding input bytes.

## Native regional ownership

The permanent interface is `ObjectSource::femMaterialIndices`, one
index into `WorldSource::materials` per authored tetrahedron, together with
`femMaterialSourceIdentity` for the source labels and declared material/density
map. `materialIndex` remains the explicitly uniform contact-interface owner.
Every selected material must have the same friction, restitution and adhesion
as that owner. Regional FEM currently rejects mixed formulation, multiphysics,
nodal field boundaries, object identification and identifiable parameters on
selected/default materials. Topology and representation remain immutable.

Each shared node receives every incident cell's density-volume contribution
once. Regional cooking sums `Float64(cooked density) * Float64(cooked volume) / 4`
in source element order before rounding nodal mass and inverse mass. Cooked
validation independently reconstructs that same sum. Empty regional fields
preserve legacy mass arithmetic. Ordinary stress, tangent, material state and
learned-material fibre lookups use the element material; no region is split
into a separate object to implement the interface.

The independent read-only review found and the native owner corrected two
restore gaps before final qualification: fixed regional parameter slots could
be replaced through snapshot environment parameters, and authored node
constraints could change while preserving mass. Restore now pins selected and
default regional parameter slots in every environment, identification routing
identities, the complete regional `restAndFixed` record, mass and inverse mass.
Unrelated identification posterior values remain restorable. A further review
found that restored topology-node lineage can change self-contact filtering.
Complete cooked topology-node records are now pinned in every framed or
regional world, together with whole immutable tetrahedron records, allocation
generation and representation. Changing a node's lineage or topology flags
cannot silently alter its source-bound contact semantics.
Complete cohesive-face, puncture-channel and topology-accounting records are
also pinned, including reserved capacity. The controls accept valid snapshots
with initially active cohesive records while rejecting invented events.
Regional scheduler base exponents retain the maximum authored material rate;
ordinary active/requested scheduler transitions remain supported.
Cooked regional constraint tags are restricted to free/static/Human attachment
values `0/1/2`. All new snapshot accesses follow exact arena-size admission.

The final [25-check run](media/cardiac-regional-material-20260912/native-runs/final-attempt-002/ctest.log)
completed in **56.61 seconds** with unchanged source hashes. It covers the new
regional compiler and Metal driver, source material/frame checks, snapshot
archives, stateful FEM/MPM, topology mutation and rollback, inverse
identification, tendon transactions and cardiac/vascular wall coupling.
The regional driver includes 40 compiler controls and three synthetic native
trajectories. Its shared-face fixture has five nodes, two tetrahedra, distinct
material IDs and test-only densities of 900 and 1,200 kg/m³. The framed source-law
trajectory takes 16 accepted steps per environment in two environments, covering
**1.6 ms**, with maximum motion **46.22 µm**. Replay and reset are bitwise;
rejected-environment rollback is isolated, and nodal mass remains unchanged.
The unframed source-law and heterogeneous-state cases each take two steps.

| Direct regional check | Relative force error | Relative tangent error | Finite-difference error |
| --- | ---: | ---: | ---: |
| Source cardiac laws | `2.37824e-7` | `2.01860e-7` | `1.65903e-3` |
| Distinct state programs | `1.51189e-7` | `2.87126e-6` | `2.06314e-5` |
| Distinct learned-material networks | `2.04093e-7` | `1.20697e-7` | `7.47584e-4` |

The direct reference gate is `3e-4`, and the finite-difference gate is `3e-3`.
Independent state projection differs by at most `6.66829e-9`; unowned state
padding is preserved. Lumped inertia differs from the FP64 element calculation
by `2.98023e-8` relative after nodal Float32 rounding. Learned-material controls
verify program and fibre routing, not trained tissue properties. The unused
fused operator helper is not executed or claimed as qualified.

The earlier two-check and 25-check passes remain as preliminary evidence before
the final topology/event guards. The corrected Python overflow-fixture failure
is retained; no native test failed in this increment.

The new format versions are Matter ABI **31**, package **16**, snapshot archive
**9** and accepted-proof manifest **9**. Historical receipts retain their
original source/runtime identities; these version changes do not requalify them.

## Remaining anatomy, density and mapping gates

The exact [source-defect evidence](media/cardiac-material-field-20260912/source-defects/REPORT.md)
still contains 31 boundary-edge defects and 47 bad vertex links. Ordinary
coordinate-preserving node separation cannot remove the remaining continuity
obstructions, and the separated RV candidate still has forbidden geometric
contacts. RA and LA have individually embedded, mutually disjoint, source-
conforming reference surfaces; distant-tissue exclusion and physical pressure
boundary admission remain open. Source artificial valve and closure structures
must retain their declared roles; they are not measured functioning leaflets.

The source mesh is loaded end-diastolic geometry. Unloaded reference coordinates,
density/inertial convention, numerical support coefficients and complete
source boundary/solver inputs remain unresolved. The source RV tissue volume
`45.024903 mL` also differs from the reported CSV value `47.440658 mL`.

The native regional interface supplies a place to declare material and density
ownership; it does not supply the missing anatomical map. The current source
material programs retain unresolved density sentinels. Positive densities in
native controls are synthetic. No blood mass is added or subtracted, no
artificial closure is silently made a pressure wall, and no subject material or
pressure-volume calibration is inferred. Source activation, a Rodero/CVSim
hybrid, full-heart mechanics, whole-body coupling, standing and walking remain
unqualified by this increment.
