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
path without charging that cost to all 416 muscles. On Apple M4 Pro the final
adaptive source build took `8:01`, versus `13:46` for the all-exhaustive
baseline.

The resulting 416 records have mean NRMSE `0.06569`, median `0.02247`, and
maximum `0.56737`; 122, 88, 42, and 2 records remain above 5, 10, 20, and 50
percent respectively. The fifth lumbricals improve from `0.65814` to
`0.10156`, medial gastrocnemius from `0.48960` to `0.04772`, and lateral
gastrocnemius from `0.47833` to `0.04958`. Remaining poor fits are retained as
explicit flags, not hidden by per-muscle hand tuning. These are source-surface
fits, not anatomical measurements of fiber or tendon length.

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
continuum, stable standing, or gait. With the 2026-08-31 adaptive artifact, the
runtime whole-body unilateral-support audit remains a failed equilibrium gate,
but improves from `3.79966` to `2.93227` normalized residual RMS and from
`147.46` to `62.30 rad/s2` maximum generalized acceleration. The fifth
lumbrical's default passive force falls from `17.4896 N` to `0.01567 N`.
The next mechanics task is source-resolved fifth-ray and wrist force sharing,
followed by the remaining hand, shoulder, and lower-body residual families;
not a force clamp or another render-only correction.
