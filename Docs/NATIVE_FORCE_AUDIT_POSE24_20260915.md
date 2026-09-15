# Native whole-body force audit after pose-search extension — 15 September 2026

The physical Mac mini M4 Pro reran the source-bound whole-body certificate on
native commit `7625ec565e086faf0dcd349846dadc2d22d65e86`. The passive-coupled
pose/recruitment search now has a bounded 24-step budget; all 23 accepted pose
steps were retained by the exact objective and the source published all 128
generalized coordinates.

The machine-readable receipt is
[`media/native-force-audit-pose24-20260915/receipt-v1.json`](media/native-force-audit-pose24-20260915/receipt-v1.json).
Its six source owners reconstruct every native net row with maximum assembly
error `6.77e-14 N`. The explicit ledger tolerances are `0.05` relative closure
and `1e-3 N` absolute residual; the maximum measured residual is
`2.28e-6 N`, the maximum closure ratio is `1.56e-5`, and the static internal
certificate is `internal_balanced=true` at normalized RMS `6.51e-6`.

The support wrench remains source-bound: `952.864475185 N` against
`952.864477038 N` body weight, six active witnesses, and a `1.85e-6 N` root
residual. Native joint and child-body indices are bound for all rows, while
anatomical labels and coordinate kinds remain unavailable.

This closes the current static generalized-force and per-DoF audit subgates.
It does not qualify temporal force convergence, sustained standing,
perturbation recovery, walking, anatomical foot loading, activation
calibration, blood-to-tissue mass transfer, calibrated materials, or subject
calibration.
