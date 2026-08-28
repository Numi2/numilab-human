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
The 2026-08-28 artifact has 416 positive records; mean fit NRMSE is `0.1254`
and maximum is `2.9797`. The maximum is retained as an explicit poor-fit flag,
not hidden by per-muscle hand tuning.

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
continuum, stable standing, or gait. In particular, the current default-pose
static recruitment remains a failed gate (`14.7176` normalized residual RMS;
`42969.6` maximum generalized acceleration). The next mechanics task is a
pose-and-recruitment equilibrium solve with joint-limit/ligament constraints,
not a force clamp or another render-only correction.
