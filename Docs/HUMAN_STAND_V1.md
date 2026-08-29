# NumiLab Human Stand v1

`numi human stand` is the first persistent full-body Human dynamics path. It
keeps configuration, velocity, MyoSim activation, current-pose route force,
large-state articulated dynamics, and source foot support on Apple Metal for
each bounded horizon. The host compiles the transaction, commits one command
buffer per phase, and publishes only after the complete phase finishes; there
is no per-step CPU dynamics loop.

## Canonical inputs

- `NHRIGID2`: one 157-body, 128-DoF floating FunctionBased articulation;
- `NHMYO2`: 416 MyoSim muscle-tendon routes, source active/passive/velocity
  curves, and one appended positive compliant-architecture record per muscle;
- `NHTENDON2`: 832 explicit body-owned route endpoints, comprising 304
  four-node surface envelopes and 528 exact source-point fallbacks;
- `NHCNT1`: ten source-authored calcaneus/toe plane witnesses; and
- BodyParts3D 4.0 bone and optional named muscle surfaces for visual review.

Every production endpoint preserves its source OpenSim/MyoSim attachment
location. An admitted four-node envelope normally distributes the endpoint
load over a connected patch on one owning BodyParts3D bone. The exact toe
semantic exception distributes one lumped EDL/FDL wrench over the four named
lesser-toe distal phalanges on their shared rigid body; hallux routes stay
one-to-one. Otherwise the exact source point remains the explicit fallback.
The rejected six-endpoint Achilles
triangle candidate remains rejected because its cross-source registration
moved sites by about 49.9 mm. A visible surface touching a bone is not
substituted for the authoritative mechanical point.

## Runtime transaction

Each device step runs, in order:

1. current-pose FunctionBased kinematics and analytic point Jacobians;
2. all 416 wrapped MyoSim route evaluations, actual `J(q)v` path velocities,
   and damped backward-Euler fiber/tendon equilibrium;
3. total tendon-force `J^T` rows and deterministic all-muscle reduction;
4. `NHTENDON2` terminal-load transfer for all 832 endpoints, with force/moment
   conservation validation and an optional same-command-buffer deformable
   consumer boundary;
5. explicit activation advancement;
6. 157-body spatial-Jacobian mass assembly, gravity/gyroscopic bias, source
   passive DoF damping through a backward-Euler solve, Cholesky forward
   dynamics, and symplectic state integration; and
7. Coulomb plane support at the ten exact foot witnesses.

The distributed terminal loads are output-only with respect to the rigid-body
step: the original source-route `J^T` projection remains the sole rigid force
authority, so publishing the envelope loads cannot add direct joint torque or
double-count muscle force. A borrowed consumer may encode into the same Metal
command buffer, but it may not commit, wait, retain, or replace that buffer and
must gate physical writes on the accepted stand status.

The standing posture compiler runs before the horizon. It solves a bounded
nonnegative activation vector against the source gravity target using the
actual per-muscle Metal force rows and acceleration-weighted residuals. This is
compilation, not a host control loop. Optional assistance is a world
force/torque spring on the floating root; it never writes joint torques. The
canonical qualification follows an assisted phase with an equal-length phase
whose root assistance is exactly zero.

NHMYO2 includes the source passive curve in the live fiber/tendon equilibrium;
it is no longer removed as a standing-only bias. The standing posture compiler
also subtracts this measured passive generalized-force row before recruiting
activation. Source passive DoF damping is preserved in both the FP64 reference
and Metal paths rather than being replaced by a hidden pose drive. The current
source-default pose is nevertheless not a calibrated whole-body equilibrium:
the 2026-08-29 M4 Pro replay reports normalized static recruitment residual RMS
`12.5546`, `compiled_stand_balanced=false`, and a bounded device horizon of
only 12.8 ms. Consequently NHMYO2 tendon mechanics and deterministic execution
are admitted, but stable standing is not.

## Run

```sh
numi human stand \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones-v4/bodyparts3d-myosim-major-bones.nhbones \
  Build/numi-human-tendon-v5/numi-human-tendon-attachments.nhtendon \
  Build/myosim-fullbody/myosim-fullbody-support-contact.nhcnt \
  Build/numi-human-stand-v1 \
  --steps 64 --timestep 0.0001 --dimension 640
```

The command automatically adds the canonical BodyParts3D full-body muscle
surface payload when it exists. It renders front, oblique, side, and rear
frames with dimension-invariant camera field of view, checks a one-step FP64
reference, executes assisted and assistance-removed horizons, and requires a
bitwise replay of final q, v, stand status, terminal-load records, and
generalized-correction diagnostics. The stand path rejects NHTENDON1 rather
than silently running without per-step terminal loads.

## Final M4 Pro qualification

Runtime `45fede450ba889b8feb1df0a8330db3c31706497` executed the final v4 bone,
v7 tissue, and v5 tendon payloads on Apple M4 Pro at 1024 × 1024 with eight
temporal and eight area-light samples. The 64 assisted and 64
assistance-removed 100 µs steps produced:

- payload SHA-256 values `969974058f5121bd0ef35689bbdb78b6aa2caba31920fa52193e218ad130efd6`
  (bones), `d3f6f3501c2a48a42677cfe940d7d7001e912cc4c8ea7979d717f6a61aabfa8b`
  (tissue), and `d563b10db8d27fdbed15d8eb196f8a57c6e6844126f91944b338917582f0aa97`
  (tendon);

- 106,496 accepted tendon transfers: 38,912 four-node envelopes and 67,584
  exact source-point fallbacks;
- maximum force/moment conservation residuals of `1.72633488546e-4 N` and
  `2.44306352215e-6 N m`;
- maximum generalized correction `7.32421875e-4`;
- one-step FP64 parity errors `3.90537220115e-8` in q and
  `3.90584484736e-4` in v;
- maximum joint-equality position/velocity errors `4.93483355513e-7` and
  `4.93483385071e-3`; and
- bitwise deterministic replay.

This is a 12.8 ms transaction and render qualification, not a seconds-long
standing result. The [exact transcript](media/numi-human-distal-continuity-v4/qualification.transcript.txt)
is retained as the prior distal-chain baseline; the current
[toe-enthesis transcript](media/numi-human-toe-enthesis-v5-2048/qualification.transcript.txt)
owns the v7/v5 device, counters, residuals, and boundary string.

## Visual review

The checked 1024 px final frames are:

| View | Evidence |
| --- | --- |
| Front | [front](media/numi-human-distal-continuity-v4/fullbody/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-front.png) |
| Oblique | [oblique](media/numi-human-distal-continuity-v4/fullbody/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-oblique.png) |
| Side | [side](media/numi-human-distal-continuity-v4/fullbody/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-side.png) |
| Rear | [rear](media/numi-human-distal-continuity-v4/fullbody/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-rear.png) |

All four frames were inspected at their original resolution. They contain the
complete head-to-feet source anatomy with no camera crop. Red surfaces are the
150 imported BodyParts3D muscle meshes. Route lines are deliberately hidden in
this clean anatomy view; focused diagnostics expose them separately.
Mechanical attachment is owned by each named bone endpoint and its route
Jacobian, which transfers route tension into force and moment on that rigid
body.

The corrected hand registration preserves the exact BodyParts3D common-frame
displacement along unsupported thumb and distal-finger chains after their
parent is attachment-refined. The transformed right-thumb gaps are 0.5 and
0.3 mm, the left-thumb gaps are 0.4 and 0.1 mm, and the corrected third-to-fifth
distal finger gaps are 0.3--0.7 mm. The previous floating thumb fragments are
absent in the retained bilateral close-ups.

The [focused left-foot views](media/numi-human-distal-continuity-v4/left-foot/)
show a continuous bone chain and bilateral source tendon surfaces. Final
left/right ankle, subtalar, and MTP angles differ by at most `8.65e-4 rad`, so
the bounded replay does not exhibit a pathological left-only divergence. The
source mechanics still has only one articulated `toes` segment per foot;
individual BodyParts3D toe bones and distal red surfaces are kinematic anatomy,
not independently actuated toes or deformable tendons.

The original standing capture metrics remain in
[qualification.txt](media/numi-human-stand-v1/qualification.txt). The current
per-step transaction design is recorded in
[HUMAN_TENDON_STEP_TRANSACTION.md](HUMAN_TENDON_STEP_TRANSACTION.md); the final
device evidence is the transcript linked above.

## Historical NHMYO1 evidence and current limit

The local Apple M4 eight-step qualification (0.8 ms per phase) reported:

- 127 nonzero compiled muscle activations out of 416 evaluated routes;
- normalized static activation residual RMS `0.0332325143483`;
- ten support witnesses, six active contacts, and maximum penetration
  `7.63684511185e-08 m`;
- one-step FP64 error `7.8900139755e-09` in q and
  `7.61414882222e-06` in v; and
- bitwise replay after the assisted and zero-root-wrench phases.

Those values qualify the historical NHMYO1 short bounded transaction. They
must not be reused as an NHMYO2 stable-standing claim. The current NHMYO2 plus
NHTENDON2 probe qualifies same-command-buffer terminal-load publication,
consumer rejection rollback, no-direct-torque identity, and deterministic
execution. It does not yet
qualify a seconds-long balance controller, exact high-velocity `Jdot*v`/RNEA
bias, joint-limit constraints, general self/environment collision, passive
preload, deformable tendon/skin/organ mechanics, gait, injury prediction, or
clinical use. Those boundaries remain explicit in every native capture line.
