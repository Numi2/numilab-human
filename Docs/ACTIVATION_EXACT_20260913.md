# Exact first-order Human activation receipt

The isolated native owner branch `human-activation-20260913` adds activation
ABI 3. The Human production path now integrates the source activation state
with an exact first-order hold over the accepted source derivative. ABI 2
remains selectable for legacy callers and compatibility checks; the change
does not alter the source force, route, or tendon state owners.

The physical Apple M4 Pro source probe passes all 416 muscles: 416 initialized
fibres advance, continuation error is `5.96046e-08`, and continuation replay is
byte exact. The probe reports `metal_max_activation_step_error` of
`2.98023223877e-08` at the `100 us` fixture step. The same library passes one
prepared native Human trajectory with 16 roots per scenario, a `100 us` step,
`1.6 ms` bounded duration, and bitwise replay. The trajectory fixture contains
three tiny pelvis samples and is explicitly not anatomical tissue or sustained
behavior qualification.

The follow-up common-grid runs at `100/50/25 us` also pass their bounded
trajectory/replay checks. Activation differences decrease from
`6.3240528e-05` to `3.0636787e-05`, while maximum applied-force differences
increase from `0.3154526 N` to `0.3554077 N`. The force result is retained as a
failed convergence observation: the exact activation integrator is not the
remaining source-force convergence owner.

Evidence: [source probe](media/activation-exact-20260913/native/source-probe.log),
[prepared trajectory](media/activation-exact-20260913/native/prepared-trajectory-100us.log.gz),
[50 us trajectory](media/activation-exact-20260913/native/prepared-trajectory-50us.log.gz),
[25 us trajectory](media/activation-exact-20260913/native/prepared-trajectory-25us.log.gz),
[refinement observations](media/activation-exact-20260913/native/refinement-observations.json),
[native manifest](media/activation-exact-20260913/native/manifest.txt),
[SHA-256 manifest](media/activation-exact-20260913/native/SHA256SUMS),
[prepared package](media/activation-exact-20260913/native/prepared-100000ns.nmatterpack),
and [prepared state](media/activation-exact-20260913/native/prepared.nhinit).

Published native source revision: `aecbdcf09f962db149709ee602142d73128335b6`
(`f02914ce9634ce94cd0e9648466cb0b499770d9b` tree), branch
`human-activation-20260913`.

This receipt closes the activation integrator's first-order temporal defect and
preserves the ABI 2 path. It does not establish held-out activation or
force-length-velocity calibration, full source-force or temporal convergence,
anatomical supports/loading, two-way blood mass/momentum transfer, unresolved
material calibration, sustained standing or walking, or a complete single-male
release.
