# Offline source joint-stop reactions

Native revision `117aa90b5c28a5448d845cbde45f3de85c481cb2` replaces
coordinatewise force cancellation in the Human equilibrium compiler with a
mass-coupled unilateral reaction solve. At the same 240-sweep budget, the full
Human's normalized internal acceleration residual falls from 12.3142 to 1.5546.
At 1,024 sweeps it reaches 0.591493. Both remain above the 0.05 balance threshold.

The compiler reduces forces and inertia through the exact NHEQ1 equality
tangent, includes dependent source joint stops, and calls the existing native
cone solver. Duplicate signed inequalities share a deterministic source row.
Unit-diagonal and homogeneous force scaling handle the large passive preloads;
a separate check verifies unregularized physical complementarity. Numerical
regularization is not a tissue material parameter.

Recruitment directions use the active reaction solve's derivative. Each accepted
candidate still requires exact nonlinear muscle forces and a new reaction solve.
The compiler also checks every source joint range after equality projection and
on pose-search candidates. Invalid initial positions preserve the caller's
accepted result. Equality reactions now satisfy the full mass equation, and
dependent accelerations are lifted through their source tangent. Earlier reports
zeroed dependent accelerations and cancelled dependent force without accounting
for coupled acceleration.

| Check | Result | Scope |
|---|---|---|
| Analytic native mechanics | Coupled lower/upper stops, a releasing stop, coincident bounds, dependent and redundant constraints, exact mass equation, recruitment and replay pass | Small independently specified slider systems |
| Native regression | CTest 2/2 and existing G1 joint-limit probe pass | Mac mini M4 Pro; no hot-loop changes |
| Full Human, 240 sweeps | Residual 1.554600; 29 loaded source stops; native physical KKT 7.2592e-13 | Still `balanced=false` |
| Full Human, 1,024 sweeps | Residual 0.591493; 28 loaded source stops; native physical KKT 7.3778e-13 | Still `balanced=false` |
| Independent reaction audit | All 122 limited source coordinates and 51 equality rows pass position, reaction-sign, complementarity, force-sum, tangent and virtual-work checks | Algebraic check of emitted FP64 compiler state |
| Source primitive oracle | Current compiled pose checked against pinned MuJoCo 3.12 geometry and gravity wrench | Initial conditions only |

The [receipt](media/limit-reactions-20260908/receipt.json) records immutable native
source, input and binary hashes, exact commands, both final runs and independent
checks. Intermediate numerical failures remain in the evidence bundle. The
first full-model attempt exposed redundant rows and poor scaling; these were
repaired without loosening the native solver's default convergence tolerance.
The recruitment fixture also retains its initial underpowered-muscle failure
and the subsequent check against its analytic force target with activation
regularization explicitly accounted for.

Reproduce the recorded audits:

```sh
python3 Docs/media/limit-reactions-20260908/verify_reactions.py \
  --log Docs/media/limit-reactions-20260908/qualified-240-stdout.log \
  --output Build/offline-reactions-audit.json
python3 Docs/media/limit-reactions-20260908/test_verify_reactions.py
```

This is offline C++ initialization and diagnosis. It does not add source-compliant
dynamic joint limits to the NumanX/Metal transaction. The 6.4 ms penetration and
joint-limit failures in the [curved-support receipt](CURVED_SUPPORT_20260908.md)
remain unresolved. The identical prepared pose must still be admitted with
registered tissue into NumanX v5, then qualify joint/contact reactions and fibre
state under load before sustained standing, recovery or walking can be promoted.

The costal callback timeout remains unresolved. No long costal retry was started
while the separate NumiVivo Metal cohort occupied the Mac mini. This does not
establish contention as the cause. Calibration, performance and full-release
qualification remain open.
