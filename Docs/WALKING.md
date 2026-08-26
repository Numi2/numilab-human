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
torque. The Core reference also provides a deterministic persistent activation
update with explicit time constants. Those constants must be provenance-locked
by the later source/contact task before training.

For the pinned Rajagopal XML the optional activation-time properties are absent.
`config/opensim-millard-activation-defaults.v1.json` therefore records the
OpenSim Millard class defaults used by the activation contract. They are not
subject calibration and must remain separately identified in any policy result.

Core revision `cf7245c` adds source-mass streamed response columns for the
direct-effort temporal-cone contact graph. Its device probe reaches a real
constraint using a source-tree body plus a deliberately synthetic sphere and
plane. This closes neither a BodyParts3D foot attachment nor a walking contact
model: the temporary shapes, plane height, friction, and compliance exist only
to exercise the owner contact ABI. Registered foot colliders, calibrated
material/contact parameters, and replayed policy outcomes are still required
before a walking rollout can be claimed.

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
