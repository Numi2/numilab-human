# Native passive-equilibrium persistent stand requalification — 15 September 2026

The isolated Mac mini M4 Pro branch `ef0fc708db0f4de1a07fca426e5a415f62e9da27`
now carries the source passive joint/tissue coupling output from the static
whole-body solve into the persistent Human stand preload behind the explicit
`--persistent-source-passive-joint-tissue` option. The default path remains
unchanged unless that option is requested.

At the canonical `12.5 us` clock and 512-step horizon, the passive path closes
the complete 128-DoF static balance (`compiled_stand_normalized_residual_rms =
9.0656247312e-06`, `compiled_stand_balanced=true`) and reduces the bounded
release peak acceleration from `33.9849624634` to `4.86072206497 m/s2` against
the same current source release. Six of ten source support witnesses are
active, the compiled support load is `952.864475301 N`, penetration is zero,
root assistance is absent, tendon rollback is preserved, and the stand replay
is bitwise deterministic.

The result is a force-balance and bounded-release subgate. It uses the current
source support witnesses and a 40-entry linearized experimental upper-joint
passive coupling; those are not calibrated anatomy. The receipt therefore
keeps sustained standing, perturbation recovery, walking, anatomical foot
registration/loading, activation calibration, blood-to-tissue mass transfer,
material calibration, and subject calibration false.

The immutable machine-readable evidence is
[`receipt-v1.json`](media/native-passive-stand-20260915/receipt-v1.json), with
paired native stdout/stderr, configure/build logs, SHA-256 identities, and the
same-horizon default comparison in the adjacent evidence directory.
