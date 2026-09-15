# Native anatomical support pose-search requalification — 2026-09-15

The Mac mini owner was rebuilt from native commit `413d08aa6a4253fc09628afa9816200a0f870969` (`Retain bounded pose search for support equilibrium`). The change keeps the caller's bounded whole-body pose budget active after the geometry-certified unilateral support set is compiled. Each candidate is equality-projected and reprojected onto the loaded support manifold before recruitment is evaluated.

The accepted diagnostic run used the source-bound one-adult-male package, the ten-primitive NHCNT2 plantar candidate, source joint equalities, source passive joint tissue, a `0.0001 s` owner step, `4096` activation sweeps, and `12` bounded pose sweeps. The binary and run log are hash-bound in `media/anatomical-support-candidate-20260915/native-static-support-receipt-v2.json`; the all-128-DoF snapshot and ledger are retained beside it.

The run closes the external static hand-off: relative body-weight error is `1.86864473173e-9`, floating-root force residual is `1.78056450295e-6 N`, the active support set is geometrically admissible, and replay is bitwise. The recruitment/posture solve improves internal normalized generalized residual RMS from the v1 run's `1.00798033623` to `0.71909649283` (`4096` activation sweeps, `12` accepted pose steps, `55` accepted global-polish steps). `internal_balanced=false` remains unchanged by design.

The force ledger retains all 128 coordinates and all six source contributions. The leading independent residual is `v_006` at `-20.4440225427 N` / `-2.52362308904 m/s²`; additional rows at `v_114` and `v_100` are about `8.58 N`. These are internal muscle/equality/support rows, so the result is evidence for the next monolithic equilibrium-owner change, not standing qualification.

This requalification does not establish dynamic contact, sustained standing, perturbation recovery, walking, activation calibration, blood or tissue mechanics, fat mass, material calibration, or subject calibration. The 12.5 microsecond synchronization clock remains a separate admitted runtime property.
