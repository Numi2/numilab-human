# Coupled offline stance equilibrium

The native compiler now passes the existing **0.05 normalized acceleration RMS**
gate for the source full body. The retained clean Mac mini run uses native
[`dcbe11f`](https://github.com/Numi2/numi-lab/commit/dcbe11f89d08f22ca920b51be3401b88fbd113fc),
1,024 recruitment sweeps and 48 posture updates. It preserves source muscle
routes and force laws, NHEQ1 ideal equalities, source unilateral stops and NHCNT2
capsule/ellipsoid support geometry. Its complete result replays bitwise.

This closes the bounded FP64 offline internal-equilibrium gate. It does not
close FP32 equilibrium, source-compliant loaded NHEQ2/NHLIM1 dynamics, registered
tissue, active standing/walking, material calibration or performance.

| Clean result | Measured value |
|---|---:|
| Internal normalized acceleration RMS | 0.0081416892 |
| Maximum root acceleration residual | 0.0119273351 |
| Support force / expected weight | 952.864475 / 952.864477 N |
| Physical source-stop KKT residual | 6.7494e-13 |
| Loaded source stops | 24 |
| Accepted coupled posture proposals | 7 of 48 updates |
| Rejected numerical evaluations / support-manifold candidates | 28 / 185 |
| Independent relative mass-action / gravity discrepancy | 6.4308e-9 / 9.6345e-7 |
| Minimum source primitive gap at FP32 pose | 1.5907e-8 m |

The accepted search trace reaches RMS 0.17377 after 16 posture updates, 0.03460
after 24, 0.01459 after 32 and 0.00814 after 48. The complete replay took
316.96 seconds of wall time while unrelated GPU work continued; this is an
offline execution record, not a performance benchmark. The bundle retains
18 exploratory runs, including three failed reaction solves. Those dirty-source
experiments are labeled separately from the clean qualification.

## Implementation and acceptance

The previous compiler reached residual 0.591493 at 1,024 sweeps. Independent
coordinate recruitment and diagonal activation corrections could not resolve
coupled muscle sharing and posture. The new bound-constrained native
Gauss–Newton proposal retains cross-muscle mass/equality/limit tangents and
local derivatives of the exact static muscle law. Exact nonlinear evaluation
with a fresh physical reaction certificate decides acceptance.

Posture proposals solve coordinates and recruitment together, compare against
scalar trials, preserve loaded stop coordinates and fit loaded source contact
planes with the owning support-placement compiler. Residual scales stay fixed
within a derivative calculation. Re-recruitment starts from the candidate's
state. Root coordinates, source ranges, plane geometry, configured displacement
bounds and the balance tolerance remain unchanged.

The contact solver is unchanged: `1e-11` KKT tolerance, 200 iterations and a
separate `1e-8` physical complementarity certificate. Reactions warm-start by
source coordinate, sign and current mass scaling. Numerical candidate failures
are counted and rejected. A failed initial solve returns a typed failure and
preserves the destination. No failed numerical evaluation becomes a physical
state. Search records are optimizer history, not simulation timesteps.

The probe now accepts `--whole-body-pose-sweeps 0..256` and exports the complete
accepted trace, FP64 q/activation, FP32 activation transport, reference fibre
lengths and signed actuator forces. Default posture budget remains four; the
passing result explicitly requests 48. A budget alone never certifies balance.

## Independent evidence

The [receipt](media/coupled-equilibrium-20260908/receipt.json) hashes the raw
certificate, clean source/binary/input identities, regression logs, independent
oracle outputs and retained failed/intermediate exploration. The raw search
certificate contains every accepted objective and numerical rejection count.

The existing source reaction audit checks every limited coordinate, all 51
equality tangents, unilateral signs, complementarity, equality virtual work and
the full force sum. MuJoCo 3.12.0 independently reconstructs all ten source foot
primitives at the FP32 transported pose and checks their 18 rows and gravity
wrench. It verifies 57 source files against the pinned archive before evaluation.

A new independent source check evaluates the full source mass matrix at the
native FP64 pose. It transforms native COM-linear/world-angular free velocity
to source body-origin-linear/body-local-angular velocity, then compares
`Tᵀ M_source T a_native` with the exported native force residual. At zero
velocity it also compares `Tᵀ qfrc_bias` with the native gravity target. The
componentwise error is `abs(a-b)/(1+abs(a)+abs(b))`; fixed bounds are `1e-6` for
mass action and `2e-6` for gravity. These numerical parity bounds allow the
native payload's FP32 source-parameter rounding. They do not fit tissue material
parameters or compare dynamic muscle force with a held-out human experiment.

Four native tests pass: static support/recruitment, motor/sensor transaction,
source equalities and source limits. Analytic regressions cover coupled force
sharing, source order reversal, activation bounds, a two-coordinate posture
equilibrium, exact replay and accepted-search history. Evidence mutation tests
reject missing trace records, increasing objectives, failed replay, altered
muscle transport, invalid fibre/sign data and corrupted generalized forces.

Reproduce the artifact audit without MuJoCo:

```sh
python3 Docs/media/coupled-equilibrium-20260908/verify_receipt.py
python3 Docs/media/coupled-equilibrium-20260908/test_verify_receipt.py
```

The source oracle additionally needs the pinned checkout and MuJoCo environment:

```sh
PYTHONPATH=Build/stance-20260908/oracle-deps:src \
  Sources/myosim/checkout/.venv/bin/python \
  Docs/media/coupled-equilibrium-20260908/verify_source_mass.py \
  --certificate Build/coupled-recruitment-20260908/clean-1024-pose48-stdout.log \
  --output Build/coupled-recruitment-20260908/source-mass.json
```

## Loaded integration still required

The current v6 joint runtime initializes source q/v and zero muscle activation.
The next integration must admit the prepared state through a typed, immutable
initial-condition contract, keep source rest coordinates separate, bind exact
Human/world identities and verify tissue attachment position/velocity at that
same pose. The authored Matter world must be compiled for it. Rewriting a
receipt hash cannot perform this registration.

NHEQ1 hard-stop reactions do not transfer as externally applied loads into the
NHEQ2/NHLIM1 runtime. Its owning physical solve must establish compliant
equality/limit/contact/fibre response from the admitted state and preserve
accepted-only publication and rollback. Standing control, accepted-root
behavior metrics and the frozen 420-trial standing/recovery/walking gate follow
on that identical loaded stack. The costal callback timeout remains unresolved;
the Mac mini's concurrent NumiVivo work prevents a controlled long retry during
this qualification. No GPU speed or sustained-behavior claim is made.
