# NumiLab Human Stand v1

`numi human stand` is the first persistent full-body Human dynamics path. It
keeps configuration, velocity, MyoSim activation, current-pose route force,
large-state articulated dynamics, and source foot support on Apple Metal for
each bounded horizon. The host compiles the transaction, commits one command
buffer per phase, and publishes only after the complete phase finishes; there
is no per-step CPU dynamics loop.

## Canonical inputs

- `NHRIGID2`: one 157-body, 128-DoF floating FunctionBased articulation;
- `NHMYO1`: 416 MyoSim muscle-tendon routes and their source force parameters;
- `NHTENDON2`: 832 explicit bone-owned route endpoints, comprising 295
  four-node surface envelopes and 537 exact source-point fallbacks;
- `NHCNT1`: ten source-authored calcaneus/toe plane witnesses; and
- BodyParts3D 4.0 bone and optional named muscle surfaces for visual review.

Every production endpoint preserves its source OpenSim/MyoSim attachment
location. An admitted four-node envelope distributes the endpoint load over a
connected patch on the owning BodyParts3D bone; otherwise the exact source
point remains the explicit fallback. The rejected six-endpoint Achilles
triangle candidate remains rejected because its cross-source registration
moved sites by about 49.9 mm. A visible surface touching a bone is not
substituted for the authoritative mechanical point.

## Runtime transaction

Each device step runs, in order:

1. current-pose FunctionBased kinematics and analytic point Jacobians;
2. all 416 wrapped MyoSim route evaluations and per-muscle `J^T` force rows;
3. activation-dependent force selection and deterministic all-muscle reduction;
4. `NHTENDON2` terminal-load transfer for all 832 endpoints, with force/moment
   conservation validation and an optional same-command-buffer deformable
   consumer boundary;
5. explicit activation advancement;
6. 157-body spatial-Jacobian mass assembly, gravity/gyroscopic/body-damping
   bias, Cholesky forward dynamics, and symplectic state integration; and
7. Coulomb plane support at the ten exact foot witnesses.

The distributed terminal loads are output-only with respect to the rigid-body
step: the original source-route `J^T` projection remains the sole rigid force
authority, so publishing the envelope loads cannot add direct joint torque or
double-count muscle force. A borrowed consumer may encode into the same Metal
command buffer, but it may not commit, wait, retain, or replace that buffer and
must gate physical writes on the accepted stand status.

The standing posture compiler runs before the horizon. It solves a bounded
nonnegative activation vector against the source gravity target using the
actual per-muscle Metal force rows. This is compilation, not a host control
loop. Optional assistance is a world force/torque spring on the floating root;
it never writes joint torques. The canonical qualification follows an assisted
phase with an equal-length phase whose root assistance is exactly zero.

The imported MyoSim passive bias is preserved in the typed inspection result
but excluded from standing dynamics. At the registered v1 pose it is not an
equilibrium preload: injecting it produced roughly 24,900 maximum generalized
acceleration units even at zero activation. Activation-dependent source force
reduced the same first-step measure to about 192. Passive muscle/tendon preload
must be reintroduced only with a registered equilibrium calibration.

## Run

```sh
numi human stand \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones-v2/bodyparts3d-myosim-major-bones.nhbones \
  Build/numi-human-tendon-v2/numi-human-tendon-attachments.nhtendon \
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

## Visual review

The checked 640 px qualification frames are:

| View | Evidence |
| --- | --- |
| Front | [front](media/numi-human-stand-v1/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-front.png) |
| Oblique | [oblique](media/numi-human-stand-v1/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-oblique.png) |
| Side | [side](media/numi-human-stand-v1/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-side.png) |
| Rear | [rear](media/numi-human-stand-v1/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-rear.png) |

All four frames were inspected at their original resolution. They contain the
complete head-to-feet source anatomy with no camera crop. Red surfaces are the
150 imported BodyParts3D muscle meshes. Cyan paths are the 416 current-pose
OpenSim/MyoSim force-route centrelines; they are a diagnostic of the mechanical
path, not a claim that a resolved tendon surface was imported. Mechanical
attachment is owned by each named bone endpoint and its route Jacobian, which
transfers the route tension into force and moment on that rigid body.

The original standing capture metrics remain in
[qualification.txt](media/numi-human-stand-v1/qualification.txt). The current
per-step transaction, rollback, replay, and four-angle evidence is recorded in
[HUMAN_TENDON_STEP_TRANSACTION.md](HUMAN_TENDON_STEP_TRANSACTION.md).

## Qualified evidence and limits

The local Apple M4 eight-step qualification (0.8 ms per phase) reported:

- 127 nonzero compiled muscle activations out of 416 evaluated routes;
- normalized static activation residual RMS `0.0332325143483`;
- ten support witnesses, six active contacts, and maximum penetration
  `7.63684511185e-08 m`;
- one-step FP64 error `7.8900139755e-09` in q and
  `7.61414882222e-06` in v; and
- bitwise replay after the assisted and zero-root-wrench phases.

This qualifies a short bounded standing transaction, force ownership, support,
assistance removal, and deterministic Apple-Metal execution. The newer
NHTENDON2 qualification additionally covers same-command-buffer terminal-load
publication, consumer rejection rollback, and no-direct-torque identity. It
does not yet
qualify a seconds-long balance controller, exact high-velocity `Jdot*v`/RNEA
bias, joint-limit constraints, general self/environment collision, passive
preload, deformable tendon/skin/organ mechanics, gait, injury prediction, or
clinical use. Those boundaries remain explicit in every native capture line.
