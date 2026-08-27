# NumiLab Human visual progress

## Active full-body source validation — 2026-08-27

<p align="center">
  <img src="media/myosim-fullbody-front.png" width="32%" alt="Anterior MyoSim full-body view">
  <img src="media/myosim-fullbody-side.png" width="32%" alt="Lateral MyoSim full-body view">
  <img src="media/myosim-fullbody-rear.png" width="32%" alt="Posterior MyoSim full-body view">
</p>

These three images are retained visual-progress artifacts from the pinned
MyoHub `myo_sim` `33c89c2bde282553dde3f526768eb3bdcfaa7649` source. They
are 640 × 480 default-pose renders of the composed `myofullbody` model. The
source is Apache-2.0; its attribution and the exact pin are in
[third-party notices](../THIRD_PARTY_NOTICES.md).

| View | SHA-256 | Inspection result |
| --- | --- | --- |
| Anterior | `136d27ebbbae009997abfdd761bbb0ae2375a3f76210f0b5d8e2ff4509a09d39` | head, bilateral shoulders/arms/hands, thorax, pelvis, legs, and feet visible |
| Lateral | `a5b3a27de38d143384d623d082643a32c43ab31d37fc471b79a1229d1cb0a52a` | continuous skull–spine–pelvis–leg posture with intact foot profile |
| Posterior | `59b98823de79643b1b7f2688d2f261bb74904fbed8a2cfd7bacef3d7f16bd97a` | posterior spine, scapular region, gluteal/hamstring chains, calves, and feet visible |

The images matter because the active mechanics source now covers the whole
body rather than treating lower-body, upper-body, and neck imports as separate
animated shells. They are visual evidence of the source model only. They do
not show native Core rendering, mesh registration, skinning, contact,
locomotion, deformable anatomy, or biological validation.

## Native articulated route snapshot — 2026-08-27

<p align="center">
  <img src="media/myosim-native-articulated/myosim-fullbody-articulated-front.png" width="32%" alt="Front native articulated route snapshot">
  <img src="media/myosim-native-articulated/myosim-fullbody-articulated-side.png" width="32%" alt="Side native articulated route snapshot">
  <img src="media/myosim-native-articulated/myosim-fullbody-articulated-rear.png" width="32%" alt="Rear native articulated route snapshot">
</p>

Core `79cc34a` captured these 640 × 640 default-pose views on an Apple M4 using
`numi human myosim-native-visuals`. The command directly reads `NHRIGID2` and
`NHMYO1`, runs the Metal articulated operator, then renders the published pose
through Core's native visual renderer. The pale shapes are intentionally simple
inertial-body proxies; red geometry is 1,815 source attachment sites plus 1,432
straight route-centreline segments. Wrap traversal remains owned by the muscle
probe; the visual centreline is a compact inspection representation, not a
claim of exact tangent geometry.

| View | SHA-256 | Rendered coverage | Inspection result |
| --- | --- | --- | --- |
| Front | `b11acf05f0f6a46d1fabd5474d4db7266d431d6f8060e30ed7a74c939f6eba47` | `17,415` body / `641` site / `1,147` route pixels | bilateral torso, pelvis, legs, feet, and routed lower-body chains visible |
| Side | `ce2b2fc5d86437d0646bc6de0644cb0a9597b7b9faaed5544d3add0c3551d484` | `9,844` body / `521` site / `850` route pixels | continuous profile with shoulder, torso, pelvis, lower-leg, and foot route evidence |
| Rear | `c5067eccb567b71319f7ee49084005f114af71476d478359ccfcaf970380c91b` | `16,873` body / `1,070` site / `2,200` route pixels | posterior trunk, pelvic, calf, and bilateral route coverage visible |

This is native pose-bound visual evidence, not a human anatomy beauty render.
It proves neither BodyParts3D registration nor skin/organ deformation, live
device-buffer presentation, contact, motion, muscle-force feedback in a
rollout, or clinical validation. The tracked visual-pack manifest records the
scene provenance alongside the three frames. Its pack and manifest SHA-256
values are `633ddb213167c1cc47b733ae80d8f25a7af36d86bf830fbf67a625f16e2a8b59`
and `8d21b2f3a265285655dde72f3611891c69a187248627419dc1be2788b101734f`.

## Native mechanics progress

```text
MyoSim source composition (offline)
              |
              v
 NHRIGID2 articulated tree + NHMYO1 muscle-route payloads
              |
              v
 C++ Core: kinematics -> spatial tendon routes -> muscle force -> J^T scatter
              |
              v
             forward dynamics (FP64 reference)
              |
              +--> Metal: poses + analytic point Jacobians -> 416 routes + static force
```

The native probe at Core `86790f3` passed with:

| Native property | Measured result |
| --- | --- |
| Source bodies / Core bodies | 103 / 157 (54 exact zero-inertia serial transform carriers) |
| Configuration / velocity dimensions | 129 / 128 |
| Active muscle-tendon elements | 416 |
| Route sites / materialized wrap geometries | 1,815 / 143 |
| Default-pose position / orientation error | `6.27e-08 m` / `9.88e-08 rad` |
| Source-oracle muscle length / force error | `2.56e-08 m` / `4.89e-04 N` |
| Inverse/forward dynamics round-trip error | `4.92e-13` |
| 416-muscle 1 µs state-coupling delta | `7.15e-02` velocity / `7.15e-08` configuration |

The last row is an unconstrained FP64 sensitivity comparison against the same
passive state, not a standing, walking, contact, or physiological-stability
result.

Run that exact native path without a Python process:

```sh
numi human myosim-native-probe Build/myosim-fullbody
```

## Apple-GPU full-body mechanics progress

The same fixed source pose is checked through the native Metal
kinematics/Jacobian plus MyoSim route-force route:

```sh
numi human myosim-native-probe Build/myosim-fullbody --metal
```

On the local Apple M4, this dispatched one 157-body / 128-DoF Human and
compared all body poses plus one nonzero point query per body against Core.

| GPU parity property | Measured maximum error |
| --- | --- |
| Body position | `6.3206736356e-07 m` |
| Body orientation component | `1.42935285885e-07` |
| Point position | `6.54161804947e-07 m` |
| Analytic point Jacobian | `7.34255547086e-07` |
| Spatial-muscle path length | `7.45058059692e-07 m` |
| Static actuator force | `2.62451171875e-03 N` |
| Applied spatial wraps | `90 / 90` |

This is actual Apple-GPU articulated execution, not a compile-only claim. The
kinematics-only route admits up to 192 bodies and 160 DoF because it does not
reserve the dense mass-factor scratch space. The 128-DoF dense mass solve,
MyoSim `J^T` scatter, and forward-dynamics stages remain the CPU reference
owner; no contact or locomotion is claimed here.

## Remaining visual/mechanical steps

1. Register BodyParts3D meshes to this articulated body in an inspected shared
   rest frame; matching anatomical labels are insufficient evidence.
2. Resolve the Mortensen spine-to-`cervical_spine` rest registration and make
   an explicit MyoSim neck/head replacement decision before applying its 72
   cervical/hyoid muscle forces.
3. Promote the default native capture into a device-resident live presentation
   sidecar, then repeat the three views over a replayed physical motion.
4. Add registered anatomical colliders and calibrated contact before any
   standing or walking qualification.

This ordering keeps the Human more realistic by retaining source mechanics and
by refusing to turn a visually plausible mesh into an uncalibrated physical
body.
