# Myofascia contact certification and accepted-state rollback

The native contact and rollback regression gates pass on Apple M4 Pro. The
full Human run also passes four, eight, sixteen, and 64 steps on runtime
`63daf986c45be89236cf5523876330bba8f0a5fc`. All four runs have bitwise replay and
accepted per-step reaction audits. These are bounded transaction results, not
stable standing, sustained tissue loading, calibrated material response, or
whole-Human completion.

The runtime changes are published on its owning `coupled` branch:
[CCD and physical rollback](https://github.com/Numi2/numi-lab/commit/a5230e9)
and the [strict Metal build correction](https://github.com/Numi2/numi-lab/commit/63daf986c45be89236cf5523876330bba8f0a5fc).

## Contact failure and correction

The original failure is now localized to zero-based step 7, microtick 0,
candidate slot 1040, primitives 177 and 487, on pectoral objects 0 and 2. The
first failed narrowphase attempt retains both triangles' start and finish
coordinates, object and primitive identities, timestep, and thickness before
rollback. This diagnostic is outside physical state and latches the first
failure for that Runtime instance; a fresh instance starts a fresh capture.

The failing feature is edge 0 of the first triangle against edge 2 of the
second. A double-precision separating-plane witness bounds the entire linear
sweep above the physical thickness. The previous conservative-advancement
search nevertheless exhausted its 16 iterations before reaching the end of
the timestep. A closest-point normal alone was insufficient on Metal because
subtraction of nearby world-space closest points distorted that normal.

The production vertex/triangle and edge/edge routines now attempt a fixed
separating-plane certificate. For every vertex pair across the two convex
features, its projected difference is affine in time. If all pairwise
projections at both endpoints exceed the effective thickness plus a floating-
point guard, every convex combination stays separated throughout the sweep.
The routines try the closest-point axis and the geometric feature normals at
the sweep endpoints. Differences are formed before projection. A failed
certificate leaves the existing collision search and rejection behavior intact.

The physical contact thickness, world-coordinate distance allowance, 16-step
search limit, contact eligibility, geometry inputs, materials, and force-owner
fractions were not relaxed by this patch. The runtime base was refreshed to
`298e5f8`, which includes other upstream mechanics changes; comparison with the
older four-step transcript is therefore not a one-change physical A/B study.
The captured-pair kernel regression independently establishes the CCD fix.

## Rejection ownership

The updated upstream tendon/FEM adapter already propagates Matter rejection
into Human status 9 (`EXTERNAL_PHYSICS_FAILED`). This change adds the missing
physical restoration in the articulated owner for ordinary stand/tendon
horizons. Before a step it checkpoints:

- articulated configuration and velocity;
- the complete MyoSim muscle state, including excitation and activation;
- the contact solver vector arena, including persistent warm starts;
- accepted status and cumulative transfer counters.

After downstream reconciliation, a rejected step restores these bytes on the
same command buffer. The failure code and failing identity remain visible;
accepted counters do not include the rejected step. Derived poses, routes, and
factorizations are recomputed and are not independently published as accepted
state. The existing NumanX prepared-root protocol retains its own authority and
does not run this ordinary-horizon reconciliation path.

A host encoding failure still abandons the command buffer. A device failure
now also restores the private Human physical state; refusing to publish a host
result alone was not sufficient. Matter retains its own existing transactional
restoration. Neither owner gains a new queue or host physics loop.

## Executable evidence

`metalrobo_matter_ccd_probe` invokes the production CCD routines: 60 cases pass,
covering true crossings, within-thickness contacts, static and separating
features, coherent translation, the captured pair, and coordinate permutations
and reflections. Miss fixtures carry independent FP64 whole-sweep witnesses.

`metalrobo_human_reconcile_probe` invokes the production Matter status adapter
and Human restoration kernel: all 12 first-step/later-step cases pass. It
checks successful-state preservation, Matter failure, Human failure, missing
Matter progress, mismatched environment, incomplete Human progress, exact
restoration of physical arrays, and accepted counters.

The complete Python suite reports 114 tests: 108 pass and six optional-fixture
cases are skipped. The added receipt regression verifies retention of source
patches and untracked source, hashes both physical shader libraries, detects a
library changing during execution, and stops the horizon sequence after failure.

Evidence is retained in [media/myofascia-ccd-20260907](media/myofascia-ccd-20260907).
The older failed horizons remain in their original evidence directory.

## Bounded horizon results

| Steps | Simulated time | Minimum J | Maximum displacement | Coupled time | Process peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.04 ms | 0.997362 | 0.07491 mm | 5.442 s | 379,289,600 bytes |
| 8 | 0.08 ms | 0.991657 | 0.26740 mm | 10.882 s | 383,762,432 bytes |
| 16 | 0.16 ms | 0.973382 | 0.98673 mm | 21.681 s | 459,259,904 bytes |
| 64 | 0.64 ms | 0.786700 | 7.56460 mm | 86.950 s | 1,366,212,608 bytes |

All four horizons retain at least 3.62802 N reaction L1 in every audited step.
The 64-step process took 215.859 seconds including setup, comparisons, replay,
and rendering. Its coupled transaction runs at about 0.736 steps/s, or a
real-time factor of 0.00000736 at this timestep. This is not interactive
performance. Peak process memory grows with the batched horizon; retained
Metal allocations and the cause of that growth have not been profiled.

The 16-step runtime frame was inspected. This front-view diagnostic does not
establish multi-angle visual quality. Large native visualization packs remain
local/on the Mac mini with hashes in `retained-visual-packs.json`; tracked
receipts, logs, visual metadata, and PNGs preserve the reviewable evidence.

The contact-certification blocker is closed for this bounded input. Longer
physical durations, timestep refinement, calibrated geometry/materials, closed-
loop balance, registered support contact, and resident performance remain open.
