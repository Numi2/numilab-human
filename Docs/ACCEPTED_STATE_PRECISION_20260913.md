# Accepted-coordinate precision and native metric publication

The retained 25 µs root-9 contact failure now passes the original nonlinear
budget and tolerance. Prepared zero-input trajectories at 100, 50 and 25 µs
each complete 1.6 ms and replay bitwise on the physical Mac mini. A separate
four-root production Brain–Human–Matter run publishes exactly four accepted
metric samples. This closes the bounded failure and supplies the first native
metric producer; `runtime.precision` and `behavior.telemetry` remain incomplete
against the full [execution-registry gates](HUMAN_GAP_EXECUTION.md).

## Permanent state and ownership

The native owner retains a 48-byte root translation: immutable episode
reference, displacement and correction. Error-free sums preserve motion below
the original world-coordinate FP32 spacing. Candidate and accepted retraction,
rollback, reset, checkpoint and accepted proof all include this state. Legacy
q.xyz is the canonical rounded projection; the existing body-state ABI stays
unchanged and explicit companion arrays retain body and point position lows.

The compensated articulated path carries paired positions through rotated
anchors, chain addition, point materialization and lever-arm subtraction.
Support offsets and signed-plane evaluation share the same precise helper and
the published orientation. The final accepted q/v refreshes body kinematics
before post-dynamics publication. Legacy articulated callers retain their
original kernel entry points and explicit FP32 mode.

Admission rejects absent, truncated, miscounted, overlapping or incorrectly
addressed companion resources. Jacobian-only queries retain null point-output
semantics. Owner authority retention includes the previous q/v/muscle
checkpoints and factor arena as well as the new coordinate buffers. Negative
controls verify rejection before encoding changes owner bytes or callbacks.

Compatibility is Matter ABI **34**, package **18**, snapshot archive **12** and
accepted-proof manifest **12**. ABI 33 packages require recooking. `NHINIT2`
preserves the version-1 byte representation and adds a version-2 exact
nanosecond timestep plus the 48-byte root state. The current transaction,
substep and Brain clock contracts still require integer microseconds:
12,500 ns can be authored but is explicitly rejected by this runtime. A
versioned end-to-end clock migration is required before the 12.5 µs trial.

## Physical execution evidence

All runs use the isolated Mac mini checkouts on Apple M4 Pro, Mac16,11,
24 GiB, macOS 26.6 (25G72), with Metal API validation enabled. The retained
[evidence directory](media/accepted-state-precision-20260912/) includes exact
commands, source and runtime SHA-256 identities before and after execution,
process ownership, logs and failed attempts. Publication identities are in
[publication.json](media/accepted-state-precision-20260912/publication.json).

| Run | Accepted roots, including replay | Physical duration per scenario | Result | Wall time |
| --- | ---: | ---: | --- | ---: |
| `precision-100us-002` | 32 | 1.6 ms | bitwise replay | 40.511 s |
| `precision-50us-002` | 64 | 1.6 ms | bitwise replay | 79.017 s |
| `precision-25us-004` | 128 | 1.6 ms | bitwise replay | 157.476 s |
| `behavior-production-001` | 4 | 0.1 ms | four joint publications and four metric samples | 7.906 s |

The independent [trajectory verifier](media/accepted-state-precision-20260912/verify_trajectory.py)
joins every trace to its committed transaction fingerprint, physics generation
and exact timestamp. It requires complete ordered q/v, motor, activation,
fibre length/velocity, path length/velocity, applied force, tendon tension and
fibre residual records, canonical coordinate expansions and exact replay of
all bytes. Eleven negative tests include identity mutations, missing or
reordered records, signed-zero and invalid receipt flags.
Ten additional comparison controls reject changed source builds, static
inputs, solver settings and authored state; only the exact timestep and its
dependent package identities may differ across the three retained fixtures.

The 25 µs trial uses the original 16-iteration nonlinear budget and 0.005 gate.
The historical root-9 failure, intermediate root-1 admission error and later
root-5/root-6 contact failures remain in the archive. No relaxed tolerance or
controller force was used to obtain the passing cohort.

Numerical controls cover 800 translation assertions, 169 independent geometry
checks and 108 support checks. The long-chain production Metal FK fixture has
maximum position error 1.155e-14 m against its independent reference; the
support secant check has error 3.549e-7 against its unchanged 2e-4 gate.
These fixtures test coordinate arithmetic, including large world origins;
they do not qualify all joint, source-force or anatomical geometry.

The release compatibility result combines 47 passing checks in
`regression-controls-002` and all three remaining bridge checks in
`bridge-controls-001`. The first run retained 46 passes and four missing
companion bindings in two direct test fixtures; the second retained a further
missing vascular-alpha binding in the bridge fixture. These fixtures now
explicitly bind every required input in legacy mode. This changed only test
executables. The production
library and both metallibs are byte-identical across the three trajectory
runs, production metric run and final compatibility cohort.

## Accepted-root metrics

The [NHBHV1 authoring path](HUMAN_BEHAVIOR_METRIC_PROGRAM.md) binds criteria to
the exact source semantic catalog, rigid bytes and cooked body frames. The
native producer records candidate measurements privately and accumulates
them only after the actual joint `COMMITTED` publication fence. An explicit
final flush uses the existing owner queue without adding a physical step.

The real `MetalNumanXGateCRootRunner` test records four attempts, four accepted
roots, zero rejected attempts and four samples from 25,000 to 125,000 ns. The
updated native owner also measures the reset pose before root one and publishes
`initial_posture_valid` and `initial_settled` from the same source-bound
geometry path. Repeated final collection is byte-identical. Minimum source-root height is
0.9449574956 m, maximum trunk tilt 0.008372304 rad, and maximum planar pelvis
COM speed 0.0001222258 m/s. These are short-run diagnostics against authored
probe criteria. The pelvis observable is not whole-Human COM.

Sixteen Swift parser/reducer tests and 44 Human authoring/evaluator tests pass.
Missing forbidden-contact coverage, native audit coverage and complete
accepted-root SHA proof remain explicit unavailable fields. The producer emits
`full_behavior_qualified=false`; no frozen standing, recovery or walking trial
is admitted by these results.

## Remaining gates and next dependencies

The [common-grid comparison](media/accepted-state-precision-20260912/refinement-observations.json)
uses the same sixteen 100 µs observation times for both refinements. Maximum
root-axis differences decrease from 6.811e-7 m to 4.383e-7 m, but maximum
applied-force differences increase from 0.6884 N to 1.4346 N. This is an
observation, not temporal convergence. Full source-route/force precision,
the 12.5 µs clock path, registered tolerances and longer duration remain open.

Quaternion composition, joint functions and trigonometry remain FP32.
Source-force route, suffix, terminal and hood geometry need their own paired
implementation and independent source-oracle checks. FEM attachment output
nodes remain FP32. A successful coordinate helper does not qualify those
separate computations or the entire mechanical system.

The anatomical supports/loading, source activation, spatial blood mass and
momentum transfer, unresolved material inputs and calibration streams retain
their existing dependency and data gates. The prepared fixture contains only
three small FEM samples with twelve attachments. It supplies no new
anatomical tissue, physiological, standing, walking or biological validation.

The subsequent [paired source-route increment](SOURCE_ROUTE_PRECISION_20260913.md)
implements the full/suffix/terminal/hood geometry follow-up and requalifies the
three supported timesteps and real metric producer on native `53670294`.
Its full-force and clock/convergence limitations remain explicit; the records
above retain their original `b0e195e1` source and runtime identities.
