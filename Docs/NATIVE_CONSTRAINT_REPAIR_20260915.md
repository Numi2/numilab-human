# Native constraint and passive-joint repair — 15 September 2026

## Current source revisions

The native development owner is `Numi2/numi-lab`, branch
`human-native-runtime-20260915`. Current numerical source was introduced by
`6fc35892f7251835bb148f818d5005c1a10303ac` (live passive joints) and
`6a73da3cb58e8b1786dfccbd254cfd7834937074` (explicit mechanics-only mode).
The source-locked full-body execution below used
`247fa64386c3a45b5cbe11f00e760fe4e2a37a6f`.
The subsequent test-only commit
`b54898b57164af68b21a4efa424256902c83886b` extends the production-kernel
contact checks; it does not change those numerical sources.

Human's initial-reference reader was integrated in
`a27a07558ef00c77748f2a03d5080fce30f0ce77`. The corrected launch and
mechanics-summary paths were integrated in
`16d159b5845f35b221382fd52e29e8605c326923` and are covered by read-only CI at
`42168460c2e1de03c826ca078d05857620c40530`.
Do not substitute the older `coupled` branch or force-merge unrelated native histories.

## Numerical changes now implemented

Static support loads are retractable normal-impulse warm starts, not an
additional permanent applied-force term. Each free-velocity solve applies
its seed once and registers that same total impulse. Friction uses the total
solved normal load; separated speculative witnesses may approach the plane.

All authored scalar joint-limit intervals participate in the coupled solve.
Their accumulated impulses may decrease or return to zero. A row activated
by another constraint is not omitted because it was initially inactive.
Static equality and limit reactions are no longer injected permanently into
applied generalized forces. Strong equality/limit pairs use a local Schur
block formed from the existing mass-response columns, without adding
artificial compliance or a second global solver. Terminal equality
impulse/residual diagnostics follow the coupled correction.

The frozen initial passive-joint preload is now replaced by a current-state
linear law, compiled from the existing source coordinate couplings. With
stiffness K, rest coordinates r, velocity v and step h, the acceleration solve is

    (M + h D + h^2 K) a = applied_force - bias - K(q-r) - h K v

The same effective factor is used by contact, equality and limit responses.
The shared CPU/Metal program validates finite scalar-coordinate mappings,
no floating-root stiffness, consistent reference coordinates, symmetry and
positive semidefiniteness. It adds no fictitious root assistance or new
material calibration. The kernel ABI is 7. A multi-environment preload
scratch-stride error was also corrected.

The host rejects the combination of this new passive program and a configured
NumanX/Human/Matter candidate program: the matching candidate tangent is not
integrated yet. This explicit restriction must not be removed without
implementing and checking that coupling.

Trace v4 reports passive potential and its change. Its continuous muscle and
static-support-reference virtual-work fields do not measure total impulsive
contact/equality work and must not be presented as a complete energy balance.

## Initial references are not runtime reactions

The mixed record is `persistent_initial_force_reference`, schema
`numi.human.persistent-initial-force-reference.v1`. It combines the initial
Metal muscle evaluation with compiled equilibrium constraint references.
The Human converter counts `passive_force_n` once; the compiled passive
reference is non-additive. It emits
`numi.human.initial-force-reference-snapshot.v1`, deliberately rejected by the
qualified generalized-force ledger even when its reconstructed residual is zero.
Explicit DoF identifiers, duplicate records/keys, nonfinite fields and peak
consistency are checked. No source revision is fabricated from a log label.

## Launch and presentation corrections

`.numi/commands/human stand` now defaults to no root assistance, source passive
joints, 64 coupled iterations, deterministic replay and a persistent trace.
The canonical 12.5 microsecond clock, 512-step default, unspecified native
recruitment ceiling, support payload ABI and authored stance options remain.
`--assisted-diagnostic` explicitly requests the older assisted/removal diagnostic.
`--mechanics-only` skips GPU rendering, uses the native marker-input shape,
and cannot be combined with the visual mechanics overlay.

The native mechanics-only option still runs the mechanics and replay checks.
It emits `myosim_articulated_mechanics=ok` with
`rendering_performed=false` and `visual_coverage_qualified=false`.
Normal visual runs retain their rendering gates. The evidence parser accepts
exactly one marker-visual, bone-visual or mechanics-only result, preserving its
kind and rejecting ambiguous or false visual claims.

The full-body execution below calls the native binary directly without the
wrapper's additional authored support-stance options. Those are distinct
initialization paths; wrapper unit tests are not the full-body execution result.

## Verification completed

The portable suites pass 708 checks: 53 unilateral-projection, 20 paired-block
and 635 passive-law checks. Linux and Apple clang checks, production Metal
compilation and complete host translation-unit checks passed. Host syntax
checks use the eight actual CMake-owned material definitions rather than
invented substitutes. The complete Human executable target was also built.

CI run `35029886472` executes the actual `mr_numi_human_stand_step` kernel,
not a duplicate dynamics kernel. It passes 4,326 checks on an **Apple
Paravirtual device**: 3,840 passive integration checks and 486 support checks.
The small two-body/two-environment fixture tests three timesteps, passive
stiffness on/off, an independent backward-Euler result, energy and angular
momentum, cold/exact/excess support seeds, lift-off, impact and Coulomb friction.
It is neither a physical M4 qualification nor a full-human contact validation.
Artifact `10420579357` has ZIP SHA-256
`d58f32f234962a570391ea534f3333a2c053bb1bf3f46ae2d06276a18d38b9d4`.

Human read-only CI run `35029343264` passes the 111-test equilibrium/launch
suite on Python 3.11 and 3.13, plus the selected existing workspace tests.
Earlier migration run `35029172331` tested the source before committing it.
Temporary write-enabled integration workflows have been retired.

## Actual full-body result: execution advanced, standing remains failed

Run `35028587673` built the complete Human target and used a hosted macOS 26
**Apple Paravirtual device**, not the user's physical M4/M4 Pro. A device-level
Metal 4 library/pipeline preflight passed. The input repository was pinned to
`3d637e7b9b07ae01a7d69ed80ed992cea9307aca`; all five source payloads were
hash-checked. The executable SHA-256 was
`31c1e8ebe0a648289437adaab3a82a2d0b93ee16ad8be17161842f8c4b1b95d5`.

The 157-body, 128-velocity-coordinate, 416-muscle model completed one step at
12.5 microseconds with the current-state passive law, no root assistance,
zero reported penetration and bitwise replay/trace endpoint equality. The
compiled static state was balanced, with six active source support witnesses
and 952.864475301 N of static support reference load.

The dynamic result is **not standing**. The actual total normal contact
impulse was zero. The kernel's largest generalized acceleration was
826.223815918 at DoF 108; the largest velocity change was
0.0103277973831 at that coordinate. Generalized coordinates mix linear and
angular units, so that acceleration is not a uniform whole-body m/s^2 metric.
The minimum contact gap was 8.40870058028e-8 m. These data expose a mismatch
between static support reference loading and the actual initial constrained
release; they do not, by themselves, identify a unique cause.

The same executable's 100-microsecond / 64-step attempt then exceeded the
300-second process deadline before publishing a native result. Its raw logs
were retained and it was terminated. The 50/25/12.5-microsecond multi-step
members were not run after this failure. The four-member common-duration
refinement therefore remains incomplete and no long-horizon claim is made.
The retained output does not contain a stack sample proving the exact cause
of this timeout.

Artifact `10420344884` contains the full build logs, device preflight,
execution manifest, one-step stdout/stderr/trace and failed 64-step logs.
Its ZIP SHA-256 is
`613785c9501196e7813146c8bea266907f9d08894d022959ae0935dc7d852583`.
One-step stdout SHA-256:
`48f485c51cb30fb33ea08d39581ff42649b3dff6f62d45516f29ceb7239ef8ad`.
Failed 64-step stdout SHA-256:
`08a837920fb76a2131675b6f71401a1567f5ae1d392ac8f2a9bfedf7aa71e67b`.
Actions artifacts have 14-day retention; reproduce from the pinned code and
input hashes rather than treating artifact retention as permanent storage.

The preceding macOS 15 attempt built but could not load Metal 4 on its device.
The first macOS 26 visual attempt reached a terminal step then failed renderer
pipeline creation. Explicit mechanics-only execution bypasses that optional
renderer; it does not repair or qualify the renderer itself.

## Remaining numerical and scientific work

Next, reconcile the representable initial pose/contact gaps, active support
set and simultaneous equality/limit solution using the actual dynamic
reactions, not the compiled references. Resolve the bounded multi-step
execution timeout and complete same-source, common-duration state/force
refinement. Do not restore permanent reaction loads or relax tolerances to
make that comparison pass. Only then increase duration to 100 ms and seconds.

Anatomical plantar loading, the full coupled nonlinear dynamics and high-speed
bias path, measured recruitment and material/subject calibration, the
NumanX/Human/Matter passive tangent, vascular mechanical ownership, sustained
standing, recovery and walking remain open. The live linear passive law and
small production-kernel tests do not establish those capabilities. No
readiness score or scientific qualification is raised by this increment.
