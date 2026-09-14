# Native whole-body all-DoF requalification — 15 September 2026

The physical Mac mini M4 Pro reran the source-bound Human static support
certificate with the full 128-row residual dump and 960 activation sweeps.
The run is bound to native source commit `260ee02ad347a3cfb2281d6333157d084f5ba17b`
and visual-probe binary SHA-256
`288fc13eb196ea9479cc350d685dc1ead681f3c5e4edf1ba1cd34fc8a1aec912`.

Command inputs were the pinned 157-body rigid package, the pinned 416-route
muscle package, `NHCNT1` support contacts, `NHEQ1` joint equalities, the
linearized experimental upper-joint passive law, and a 100 µs solver response
step. The native output and receipt are retained in
`Docs/media/native-whole-body-all-dof-20260915/`.

The source root wrench remains closed: body mass is `97.1319506911 kg`, total
support is `952.864475184 N` against `952.864477038 N` weight, relative weight
error is `1.94529905163e-9`, and the maximum root force residual is
`1.85360568139e-6 N`. Six of ten authored support witnesses carry nonzero load,
position-limit KKT residual is `7.03787353392e-13`, and replay is bitwise.

The internal result is unchanged at the relevant precision: normalized residual
RMS is `0.141417024366`, maximum reported acceleration residual is
`0.521694103204 m/s²`, no coupled pose step was accepted, and
`internal_balanced=false`. Raising the sweep budget from the prior 128-run to
960 therefore does not solve the loaded articulated state. The residual dump is
diagnostic evidence for the next monolithic pose, activation, tendon/fibre and
contact solve; it does not qualify force convergence, dynamic release,
sustained standing, recovery, or walking.

The qualification remains **partial** by design. Root support balance and
replay are proved for this static certificate; internal generalized balance,
activation calibration, anatomical contact material, and subject calibration
remain open.

The same raw reaction record is now canonicalized into a six-owner force
snapshot and per-DoF ledger by `equilibrium_force_snapshot` and
`force_ledger`. The immutable artifacts are retained as
[`force-snapshot-v2.json`](media/native-whole-body-all-dof-20260915/force-snapshot-v2.json),
[`force-ledger-v2.json`](media/native-whole-body-all-dof-20260915/force-ledger-v2.json),
and [`receipt-v2.json`](media/native-whole-body-all-dof-20260915/receipt-v2.json).
All 128 rows reconstruct the native net to `1.1368683772161603e-13 N`, but the
maximum normalized closure ratio remains `1.0` (internal and root), with RMS
closure ratio `0.22314108914802203`; the ledger therefore remains partial.
The worst ranked coordinates are `v_051`, `v_089`, `v_005`, `v_102`, `v_058`,
and `v_096`, dominated by gravity, muscle-tendon, and support owners. This
makes the remaining generalized-force failure directly consumable by the next
monolithic pose/contact/tendon solve.
