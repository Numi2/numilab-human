# Active muscle control: 8 September 2026

Numi Human now has an explicit source-bound Brain/Metal controller for tonic
spindle feedback and authored periodic recruitment. This is a bounded active
control increment. Sustained standing, recovery, walking and the complete
neuromusculoskeletal release remain unqualified.

## Native corrections

The NHMYO2 activation kernel treated fibre length and velocity as legacy reserved
zeros. After the first step initialized fibre length, subsequent steps silently
retained the prior activation. The new continuation regression fails against the
old shader. The corrected shader advances initialized fibres, retains the
zero-length sentinel constraint, and continues source-owned fibre/force evaluation.
On the M4 Pro, all 416 initialized activations advance with zero error against the
source derivative and byte-identical replay. No extra generalized force or root
assistance is introduced.

The public tendon-tension receptor also incorrectly returned tension normalized
by maximum isometric force while declaring newtons. It now publishes positive
source path tension in newtons. The native IO probe checks a force scale different
from one, and passes replay, failed motor authority, reserved publication and
terminal quarantine. Historical sensor records before this correction must not be
reinterpreted as newton measurements. For compliant NHMYO2, the legacy field named
applied active force contains the applied tendon force, including passive
contributions; it does not isolate contractile force.

## Source-bound authoring and execution

The new command is:

```sh
numi human locomotor-program body.json myosim.nhmyo standing.json \
  --tonic 0.03 --length-gain 0.4 --velocity-gain-seconds 0.02
```

`body.json` comes from `numi-brain-gate-c describe-body` with the actual native
asset/world arguments. Its model and sensory fingerprints and muscle SHA-256 must
match the payload. Each source reference path length is retained individually.
The three gains above are explicitly authored research values, not experimental
calibration results. The program's `calibrationArtifactSHA256` currently binds
those source lengths, not independent gain identification.

An optional `--period-microseconds` and `--gait-map` admit explicitly indexed
sine/cosine recruitment coefficients. No bilateral role or gait is guessed from
an actuator number. Phase follows committed physical time. The Brain Metal 4
controller uses only delivered path-length and path-velocity receptors; both
feature-validity bits are required. It rejects competing controllers/goals,
authenticates its identity in checkpoints, and enters the existing protected
muscle output before native force evaluation. Missing observations produce zero
command, including the initial bootstrap. Native anatomy no longer inherits the
six-channel transport fixture's tonic and reflex routing.

Use `numi-brain-gate-c capture --muscle-locomotor-program standing.json` with the
same native arguments and explicit research dataset coordinates. Captures retain
exact motor and root artifacts with `promotable=false`. They cannot issue legacy
Gate C qualification or train an unrelated policy. The full command invocations,
programs, logs and source/binary identities are retained with the receipt below.

## Evidence and remaining work

The 100-microsecond native controller test runs four roots, exact replay and a
proprioception ablation: 12 accepted roots across three runs. It requires actual
activation persistence and a musculotendon force response, in addition to
command/sensor replay. Peak command is 0.0008202028; peak activation reaches
0.000016390819 and remains positive while excitation returns to zero. Strong
joint-risk inhibition remains visible. This is 0.4 milliseconds per run and is
not evidence of sustained balance.

The costal v5 capture uses the registered mass-conserving world and source joint
equalities at 10 microseconds. Its first attempt hit the 30-second callback
budget; the final runner uses the existing costal integration test's bounded
60-second budget. Timing is retained as offline completion evidence, not
interactive performance. See [the receipt](media/active-control-20260908/receipt.json)
and [the control verification](media/active-control-20260908/costal-control-verification.json)
for final attempted/accepted roots, replay, dropout and emergency-stop results.

All four runs accept four roots each (16 total, 40 microseconds per run).
Active excitation after bootstrap is 0.013320053–0.013493109. Valid settled
activation reaches 0.000053751857 in the input to the fourth root. Motor and
all seven sensor modalities replay exactly. Dropout and emergency stop produce
zero excitation; valid emergency-stop observations retain zero activation.
Bootstrap values have zero validity and are not counted as physical observations.
No force or kinesthetic difference is resolved in these valid settled samples
against emergency stop. Costal force-response qualification therefore remains
false even though the activation/transaction checks pass.

The earlier eight-step standing baseline predates the activation fix and reports
`balanced=false`; it used the
legacy NHEQ1 visual probe. Source NHEQ2 input is rejected by that reader and those
failed attempts are retained. It is not a costal/source-compliant standing result.
The periodic probe is a single-channel transport experiment; numerical phase
alternation is tested separately from physical gait.

A fresh check on the corrected native shader also reports `balanced=false`.
It runs the native recruitment baseline (not the Brain program), with NHTENDON3,
legacy NHEQ1, no root-assistance flags, eight 100-microsecond steps and exact replay.
Its maximum velocity delta is 205.933868408 and minimum support gap is
-0.0140668628737 m; normalized static residual RMS is 38.3838027852. These are
negative stability results, despite successful runtime completion. The four native
512-pixel views were inspected: the skeleton is rendered intact, but a short
static view is not balance evidence. This invocation differs from the earlier
baseline and is not a controlled before/after stability comparison.

The existing costal v5 joint-publication regression also passes after both native
corrections (275.121 seconds test wall time). It covers eight accepted roots,
rollback/retry and the retained deterministic learning checks. Local Python
validation runs 178 tests with seven skips and no failures. Mac mini's targeted
Brain/controller, body-contract and decoder-fork suite runs 21 tests with one
released-data skip and no failures; the native physical controller test runs
separately with the actual bridge paths configured.

Next completion dependencies are an equilibrated anatomical initial posture and
loaded contact, experimentally supported recruitment/reflex gains, calibrated
joint-state risk, native accepted-root behavior reductions, and the frozen
420-trial standing/recovery/walking evaluation. Longer source/timestep/mesh
convergence and all five performance envelopes remain required. Source admission,
short active recruitment and replay do not close these rows.
