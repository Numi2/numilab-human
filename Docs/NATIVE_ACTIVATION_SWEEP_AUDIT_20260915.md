# Native activation sweep audit — 15 September 2026

The physical Mac mini M4 Pro ran the whole-body support-wrench mode for the
one-adult-male source package at the canonical 12.5 µs clock. The run evaluated
32 global activation candidates, accepted 24 activation polish steps and four
pose steps, and replayed bitwise.

The floating-root numbers close: body weight is 952.864477038 N, the selected
support total is 952.864475255 N, relative weight error is
`1.87096257887e-9`, and maximum root-force residual is `1.78277309715e-6 N`.
The articulated state does not close. The internal normalized residual RMS is
`17.1808154012`, `internal_balanced=false`, and the highest ranked coordinate is
DOF 119 with acceleration residual `90.9280057267` and muscle-force term
`-121.819171408`. Nine of ten authored support witnesses are active in this
static diagnostic.

This is a hard negative result for the current standing path. A root wrench or
body-weight match cannot be promoted to whole-body equilibrium, activation
calibration, anatomical contact loading, material calibration, blood mechanics,
standing, recovery, or walking. The source checkout used for this diagnostic
was detached and dirty; the receipt records the source commit, binary hash and
every payload hash so it cannot be confused with the clean production owner.

The next solver change is therefore constrained: solve the ranked internal
coordinates and their passive/tendon/contact terms in the coupled equilibrium,
then rerun the same sweep and require the internal residual gate to close before
any long-horizon behavior run.

Receipt: `Docs/media/native-activation-sweep-20260915/receipt-v1.json`.
