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
             forward dynamics
```

The native probe at Core `b2d4490` passed with:

| Native property | Measured result |
| --- | --- |
| Source bodies / Core bodies | 103 / 157 (54 exact zero-inertia serial transform carriers) |
| Configuration / velocity dimensions | 129 / 128 |
| Active muscle-tendon elements | 416 |
| Route sites / materialized wrap geometries | 1,815 / 143 |
| Default-pose position / orientation error | `6.27e-08 m` / `9.88e-08 rad` |
| Source-oracle muscle length / force error | `2.56e-08 m` / `4.89e-04 N` |
| Inverse/forward dynamics round-trip error | `4.92e-13` |

Run that exact native path without a Python process:

```sh
numi human myosim-native-probe Build/myosim-fullbody
```

## Remaining visual/mechanical steps

1. Register BodyParts3D meshes to this articulated body in an inspected shared
   rest frame; matching anatomical labels are insufficient evidence.
2. Resolve the Mortensen spine-to-`cervical_spine` rest registration and make
   an explicit MyoSim neck/head replacement decision before applying its 72
   cervical/hyoid muscle forces.
3. Add Core-native presentation for the full-body payload, then repeat these
   three views from the native runtime.
4. Add registered anatomical colliders and calibrated contact before any
   standing or walking qualification.

This ordering keeps the Human more realistic by retaining source mechanics and
by refusing to turn a visually plausible mesh into an uncalibrated physical
body.
