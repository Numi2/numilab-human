# Compliant muscle-tendon v2

`NHMYO2` adds a live force-transfer law to every one of the 416 MyoFullBody
routes. BodyParts3D remains the anatomical surface source; MyoFullBody remains
the full-body route and force-curve source; the compliant architecture is an
explicit inference where the free source data does not identify one.

## Offline inference

The compiler retains each actuator's authored operating length interval,
active-force-length curve, passive curve, velocity curve, maximum force,
route sites, and wraps. It then searches positive optimal-fiber length `L0`
and tendon-slack length `LT` against the actuator's own static force surface,
including its default-pose length. `L0` and `LT` are searched independently:
many rigid-tendon source actuators contain an offset that would imply negative
slack if copied algebraically.

Every appended 32-byte record contains `L0`, `LT`, tendon strain at normalized
force one (`0.049`), stiffness (`1.375 / 0.049`), toe-end force (`2/3`),
curviness (`0.5`), normalized fiber damping (`0.1`), and fit NRMSE. Pennation
is zero because it is not identifiable from the retained free full-body source.

The 2026-08-31 fitter searches dimensionless positive `L0` and `LT` ratios
over a wider domain. A deterministic 21-by-21 pass classifies each source
surface. Fits above five percent NRMSE receive a 41-by-41 global search and
three local 21-by-21 refinements; already-good fits use two local 11-by-11
refinements. This keeps narrow hand, calf, and axial minima on the exhaustive
path without charging that cost to all 416 muscles.

The first adaptive objective sampled activations `0.1`, `0.5`, and `1.0`.
That was insufficient: large active forces could hide a false force at rest.
The passive-aware objective now also samples activation zero and weights that
channel by `1024`. This is a mechanical gate rather than a visual heuristic:
force present with no activation enters every subsequent equilibrium solve.
The manifest records the source and fitted passive force at the source pose
for every muscle, plus aggregate absolute-error statistics.

On Apple M4 Pro the passive-aware 416-muscle source build took `16:28`. Its
weighted fit NRMSE is mean `0.09582`, median `0.04276`, and maximum `0.58500`.
Those values are not directly comparable to the former active-only NRMSE
because the objective changed. The directly comparable passive-oracle error
improves from mean `3.5242 N`, p95 `15.4167 N`, and maximum `137.6033 N` to
mean `0.81787 N`, p95 `4.44320 N`, and maximum `29.38435 N`. Counts above
`0.1`, `1`, and `10 N` fall from `142/82/34` to `100/54/8`.

For the fifth interosseous `UI_UB5`, the former architecture predicted
`2.2306 N` at activation zero against a source value of `0.07459 N`; the new
fit predicts `0.21346 N`. The fifth lumbrical, radial interosseous, and fifth
superficial flexor fits reproduce their zero source passive force to numerical
precision. The soleus worst-case error falls from `137.60 N` to `25.43 N`.
Remaining errors are retained as explicit flags, not hidden by per-muscle hand
tuning. These are source-surface fits, not anatomical measurements of fiber
length, tendon slack length, or pennation.

## Runtime law

For each step, Metal evaluates the current wrapped path and `J(q)v`, advances
activation, and solves zero-pennation fiber/tendon equilibrium with damped
backward Euler. The accepted fiber length and velocity are persistent sidecar
state. Tendon tension is the only force projected through the route Jacobian.
`NHTENDON2` then publishes equal-and-opposite terminal loads to the registered
bone point or connected four-node surface envelope. These loads are available
to bone/tissue consumers but are not re-added as joint torque.

The CPU double-precision implementation is the executable oracle for the
Metal implementation. Both reject nonpositive architecture, nonfinite state,
or an invalid path. Startup fiber length is always clamped inside the actual
musculotendon path, including muscles whose inferred `L0` exceeds that path.

## Current evidence boundary

On Apple M4 Pro with Metal API validation enabled, the 416-muscle reference
probe passed with maximum normalized tendon tension `0.726742`, normalized
equilibrium residual `0.006543`, muscle-force error `0.152347 N` on a
`1238.398 N` reference scale, and bitwise tendon replay. Across 832 terminal
loads, maximum force residual was `3.74e-05 N` and maximum moment residual was
`1.27e-06 N m`.

This qualifies the numerical muscle/tendon transaction and explicit
tendon-to-bone load transfer. It does not qualify anatomical pennation,
subject-specific slack lengths, enthesis material failure, a deformable tendon
continuum, stable standing, or gait. With the passive-aware artifact, the
runtime whole-body unilateral-support audit remains a failed equilibrium gate,
but improves from the prior adaptive artifact's `2.93227` to `2.58813`
normalized residual RMS and from `62.30` to `30.59 rad/s2` maximum generalized
acceleration. Bilateral fifth-MCP abduction falls from `62.30/60.34` to
`1.76/1.25 rad/s2`; no coordinate exceeds `100 rad/s2`, five exceed
`10 rad/s2`, and 52 exceed `1 rad/s2`. Body weight closes to `2.26e-9`
relative error and replay is bitwise.

The next mechanics task is source-resolved bilateral wrist and third-MCP force
sharing, followed by the remaining shoulder and lower-body residual families;
not a force clamp, invented rest torque, or render-only correction.
