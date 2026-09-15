# Native constraint repair — 15 September 2026

## Source revisions

The corrected physics owner is `Numi2/numi-lab`, branch
`human-native-runtime-20260915`, commit
`4956a2858620ec60be34f1a2333a3bfaf43e1194`. The actual block-kernel integration
is `cbed82bf12791e89a19ffe86c188df3a0493fa33`; the following commit retires
one-shot migrations and makes the source CI read-only. Do not substitute the
older `coupled` branch or force-merge unrelated native histories.

Human's reusable refinement reader and new run binding were published on
`main` in `599ad6cb65e6314a5dabaacb6ffe80d4b9087ec5`.

## Numerical changes

Static support loads are now retractable normal-impulse warm starts, not a
second permanent applied-force term. Every new free-velocity solve applies
its seed once and registers that same total impulse. Friction uses the total
solved normal load; a separated speculative witness may approach the plane
instead of being forced to stay still above it.

All authored scalar joint-limit intervals participate in the coupled solve.
Their accumulated impulses may decrease or return to zero. A row activated
by another constraint is no longer omitted merely because it was inactive
at the initial free velocity.

Static equality and joint-limit reactions are no longer permanently injected
through the runner's generalized-force preload. The mixed initial muscle/
static-reaction record is explicitly named `persistent_initial_force_reference`,
not `persistent_dynamic_force_audit`. It is not a runtime reaction measurement.

Strong equality/limit pairs use a local Schur block formed from the existing
factored mass-response columns. This targets the near-dependent rows in the
published refinement trace without a second global solver or artificial
compliance. Equality impulse/residual diagnostics are sampled after the
terminal coupled correction rather than an earlier intermediate state.

## Verification completed

The shared C++ policies pass 53 unilateral-projection checks and 20 paired
block checks on Linux and Apple clang. The production Metal kernel compiles
with Apple's Metal compiler. Native CI run `35020445714` completed both jobs.
These checks do not run a full Human GPU trajectory.

Human CI run `35019892381` passes 75 tests on each of Python 3.11 and 3.13,
including historical refinement, force ledgers, state handoff, and 17 new
run-binding regressions. The new cases reject changed source/binary/payloads,
incomplete runs, duplicate/nonfinite metrics, and assistance-enabled manifests.
Zero acceleration peaks no longer cause division by zero.

## Physical requalification still required

No physical M4 full-body result has been produced for this corrected owner.
Do not transfer earlier acceleration, contact, timing, or standing claims to it.
The initial passive-tissue preload remains a bounded-release approximation;
it must become a current-state force law before long-duration tissue claims.
Existing virtual-work summaries containing static support references are not
measurements of work done by the new impulse-only support solve.

Rebuild the pinned native owner and repeat the same 6.4 ms duration at
100/50/25/12.5 microseconds with 64/128/256/512 steps, no root assistance,
fixed source payloads, and deterministic traces. Compare same-time q/v,
post-projection contact/equality/limit residuals, solved reactions, and work;
matching acceleration peaks alone must not qualify force convergence.

For a new evidence directory, use the existing reader with a run manifest:

```sh
PYTHONPATH=src python3 -m numilab_human.native_passive_stand_refinement \
  --case-root /path/to/new-native-cases \
  --run-manifest /path/to/new-native-cases/run-manifest.json \
  --output /path/to/new-native-cases/refinement.json
```

The manifest schema is `numi.human.native-refinement-run.v1`. It records the
full native commit, clean-worktree declaration, device, no-assistance policy,
and hashes/paths for the binary plus rigid, muscle, tendon, support, and equality
inputs. Each named case binds its actual exit code, grid, artifact hashes,
and stdout/stderr hashes. The executable fixture in
`tests/test_native_refinement_binding.py` specifies the format; its synthetic
files are tests, not physical evidence. New cases cannot inherit the historical
source or binary identity by default.

Only after full state/force refinement should release duration advance to
100 ms and controlled standing to seconds. Anatomical contact admission,
state-dependent tissue mechanics, measured muscle/material/subject calibration,
vascular mechanical ownership, sustained standing, recovery, and walking remain
open. No readiness score or qualification gate is raised by these code checks.
