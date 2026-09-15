# Native force-audit requalification — 15 September 2026

The physical Mac mini M4 Pro reran the source-bound whole-body support
certificate on native commit `ef0fc708db0f4de1a07fca426e5a415f62e9da27`, with
the `12.5 us` response step, 960 activation sweeps, the 40-entry source
passive joint/tissue coupling, and the complete residual dump for all 128
velocity coordinates.

The run is retained in
[`media/native-force-audit-20260915/receipt-v1.json`](media/native-force-audit-20260915/receipt-v1.json).
The receipt binds the M4 Pro binary, five source payload hashes, native stdout,
the canonical six-component force snapshot, and the per-DoF ledger. Every row
contains gravity, muscle/tendon, joint-equality, joint-limit, support, passive
tissue, and authoritative net values. Reconstruction error is
`1.1368683772161603e-13 N`.
The companion coordinate map binds all 128 rows to native joint and child-body
indices; anatomical labels and coordinate kinds remain explicitly unavailable.

The root support wrench remains closed: body mass is `97.1319506911 kg`, total
support is `952.864475184 N` against `952.864477038 N` weight, relative weight
error is `1.94529905163e-9`, and maximum root force residual is
`1.85360568139e-6 N`. Six of ten authored support witnesses carry nonzero load;
the position-limit KKT residual is `7.03787353392e-13`, and replay is bitwise.

The internal result remains open. Normalized residual RMS is `0.141417024366`,
the maximum per-coordinate closure ratio is `1.0`, and `internal_balanced=false`.
The ranked ledger identifies the remaining coordinates and dominant owners
without hiding them behind the balanced floating root.

This closes the current diagnostic coverage gate only. It does not qualify
dynamic force convergence, a 100 ms release, sustained standing, perturbation
recovery, walking, anatomical support/loading, activation calibration,
blood-to-tissue mass transfer, material calibration, or subject calibration.
