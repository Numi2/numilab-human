# Continuous-walking engineering contract

`numi human walking-contract` is the hand-off boundary for a learned
lower-body walking task. It reads the pinned Rajagopal source and emits the
exact six `ground_pelvis` coordinate order, all 35 source coordinates, and the
complete ordered 80-muscle excitation action surface.

```sh
numi human walking-contract --sources Sources --output Build/rajagopal-walking-contract.json
```

The contract deliberately rejects a substituted floating base or an incomplete
muscle set. Core must execute the source `ground_pelvis` FunctionBased joint as
a mobile root; a policy action is a bounded excitation, not a direct joint
torque. The bounded fixed-root Core path can now consume that complete action
surface through a native task bridge and execute deterministic device activation
updates with explicit time constants. Mobile-root task and policy execution
still needs its own admission before training.

For the pinned Rajagopal XML the optional activation-time properties are absent.
`config/opensim-millard-activation-defaults.v1.json` therefore records the
OpenSim Millard class defaults used by the activation contract. They are not
subject calibration and must remain separately identified in any policy result.

Core revision `5e46d13` adds a fail-closed native-task bridge alongside the
packed per-control excitation stream. It admits exactly one ordered
Millard-excitation action per source muscle, rejects mixed generic actuation or
partial action surfaces, maps the conventional signed task action from `[-1, 1]`
to excitation `[0, 1]`, and applies the same device first-order activation
update before source force projection. Its local Apple M4 smoke probe evaluates
all 80 source muscles and reaches a real constraint using a source-tree body
plus a deliberately synthetic sphere and plane. Full explicit excitation and a
complete native task action surface both produce larger device force than the
source-default state without CPU force restaging; final device activation values
match the exact first-order reference. This closes neither a BodyParts3D foot
attachment nor a walking contact model: the temporary shapes, plane height,
friction, and compliance exist only to exercise the owner contact ABI.
Registered foot colliders, calibrated material/contact parameters,
mobile-root source dynamics, deterministic reset/replay, and policy outcomes
are still required before a walking rollout can be claimed.

The flat-ground walking scenario is blocked until the following artifacts are
present and validated: per-foot BodyParts3D-to-Rajagopal registration,
conservative collision proxies, friction/compliance parameters, collision
exclusions, and a deterministic reset/replay scenario. It is likewise invalid
to animate an anatomical layer until every mesh has a validated segment
attachment; unregistered geometry stays hidden or static.

This contract is not a Human RobotPack, trained policy, calibrated gait model,
OpenSim parity result, or tissue-physics implementation.

## Attachment and contact worklist

```sh
numi human attachment-worklist --sources Sources --output Build/lower-body-attachments.json
```

This emits review candidates from the original BodyParts3D labels for the
Rajagopal pelvis, legs, and feet. It also records the four source foot bodies
that need conservative collision proxies. Candidate names are not transforms:
every entry must be registered and visually reviewed before animation or
collision use.

For the four walking-contact bodies, produce the stricter, provenance-pinned
handoff before authoring a collider manifest:

```sh
numi human foot-registration-template \
  --sources Sources \
  --output Build/foot-registration-template.json
```

The template enumerates `calcn_r`, `toes_r`, `calcn_l`, and `toes_l`, retains
the BodyParts3D archive hashes and Rajagopal XML hash, and lists only
side-specific review candidates. It intentionally contains no transform,
proxy, pair exclusion, or material value. Each entry must receive a reviewed
source-to-body rest transform, multi-angle landmark/residual evidence, a
conservative collision proxy, and a contact-calibration receipt before it can
be converted into a walking task artifact.

The source-local proxy preflight makes the geometry side of that review
reproducible without claiming a collision binding:

```sh
numi human foot-collider-preflight \
  --sources Sources \
  --output Build/foot-collider-preflight.json
```

It emits the exact OBJ hashes, source-millimetre bounds, and a conservative
axis-aligned enclosure for every laterality-qualified foot mesh. The boxes do
not have an OpenSim transform, contact pair, or material; they become eligible
for conversion only after the corresponding reviewed registration receipt.
