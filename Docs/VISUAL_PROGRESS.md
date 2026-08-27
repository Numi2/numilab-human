# NumiLab Human visual progress

## Presentation correction — 2026-08-27

The earlier native route galleries below are retained only for renderer
regression and source-coverage evidence; they are **retired as anatomy
presentation**. They drew every MyoSim route as a straight line between sites
and wrap centres, which can cut through a sphere/cylinder wrap and visibly miss
an anatomical surface. They are not acceptable tendon imagery.

The current native renderer keeps routes hidden by default. Its focused
1024 × 1024 inspection mode starts and ends at exact source sites, uses the
source solver's tangent contacts, and samples the wrapped sphere/cylinder arc.
The Numi route-inspection command projects only those authored source-site
endpoints to their nearest matching BodyParts3D bone triangles for visible
attachment; it never changes the underlying force path. The BodyParts3D
major-bone payload also has a visual-only per-bone refinement against exact
MyoSim attachment sites. These are alignment diagnostics, not tendon-surface
geometry, an attachment transfer, or a medical registration. Focused
replacement captures remain outside the showcase until review.

## Reviewed focused calcaneal-tendon source inspection — 2026-08-27

<p align="center">
  <img src="media/bodyparts3d-right-calcaneal-tendon-2048/right-lower-leg-anterior.png" width="32%" alt="Right posterior lower-leg source anatomy, axis-negative-y camera" />
  <img src="media/bodyparts3d-right-calcaneal-tendon-2048/right-lower-leg-oblique.png" width="32%" alt="Right posterior lower-leg source anatomy, oblique camera" />
  <img src="media/bodyparts3d-right-calcaneal-tendon-2048/right-lower-leg-posterior.png" width="32%" alt="Right posterior lower-leg source anatomy, axis-positive-y camera" />
</p>

These are reviewed native 2048 × 2048 reference frames of a deliberately
uncluttered, exact BodyParts3D 4.0 source-rest-frame bundle: right tibia,
fibula, talus, calcaneus, lateral and medial gastrocnemius, soleus, and the
right calcaneal tendon. The importer preserves every compatible authored OBJ
vertex normal (eight of eight selected surfaces) rather than replacing the
source shading with generated normals. The tendon is a distinct gold semantic
layer, muscle is red, and bone is ivory; those colours are presentation labels,
not tissue parameters.

The offline source check finds exact nearest-vertex separations of `0.828009`
mm from calcaneal tendon to calcaneus, `0.143963` mm to lateral gastrocnemius,
`0.398794` mm to medial gastrocnemius, and `0.248701` mm to soleus. That makes
the selected source surfaces visibly continuous in this cutaway; it does not
prove a watertight biological attachment, a MyoSim registration, a force-path
transfer, or a mechanical tendon constraint.

Core `e028486` cooked the 13,349-vertex / 20,642-triangle GLB (three semantic
materials) and rendered each camera with an independent native reference
workspace on the local Apple M4. The Mac mini was occupied by an unrelated
BirdFlow training run, so these are a bounded local-GPU fallback rather than
an M4 Pro qualification. Each camera had nonzero source segmentation and its
exact capture metadata is retained in the [transcript](media/bodyparts3d-right-calcaneal-tendon-2048/capture.transcript.txt).

| Camera | PNG SHA-256 | Source pixels |
| --- | --- | ---: |
| `axis_negative_y` | `4effa463ec9034bd1dd5d730d119132af0d1bad9dfe370cb010d54c9e74979f0` | 295,907 |
| `oblique_positive_x_negative_y` | `04cb010ccb88e2e8fba6bc0c2deb85e78b8707fd43179fbd25198e77f31c3b7a` | 239,767 |
| `axis_positive_y` | `40b3a2501ce4e8eee597ea143b9fe98eb94c8a990f5f82a7c9998e21e84b9c5a` | 311,627 |

This is the current source-anatomy presentation. It is not a full Human beauty
render, skin, organ/vessel/nerve view, articulated deformation, muscle-driven
rollout, or medically validated attachment model. Those claims remain gated by
their own geometry and mechanics evidence.

## Retired full-body source validation snapshot — 2026-08-27

These images are retained source-provenance artifacts from the pinned MyoHub
`myo_sim` `33c89c2bde282553dde3f526768eb3bdcfaa7649` source. They are 640 ×
480 default-pose renders of the composed `myofullbody` model, and are retired
from presentation because their framing and generic source geometry do not
make a credible anatomical tendon view. The source is Apache-2.0; its
attribution and exact pin are in [third-party notices](../THIRD_PARTY_NOTICES.md).

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

## Retired native articulated route snapshot — 2026-08-27

Core `79cc34a` captured these 640 × 640 default-pose views on an Apple M4 using
`numi human myosim-native-visuals`. The command directly reads `NHRIGID2` and
`NHMYO1`, runs the Metal articulated operator, then renders the published pose
through Core's native visual renderer. The pale shapes are intentionally simple
inertial-body proxies; red geometry is 1,815 source attachment sites plus 1,432
straight route-centreline segments. It remains a historical coverage artifact,
not a tendon rendering or a path-to-bone attachment assessment.

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

## Retired native BodyParts3D 27-major-bone route-overlay snapshot — 2026-08-27

Core `2aab522f92f44644c35bbde1a8ea3fd85356b027` captured this exact
`NHBONES1` package on the Apple M4 Pro on `macmini`. The native C++ program
read the compiled `NHRIGID2`/`NHMYO1` payloads plus 27 source-derived
BodyParts3D major-bone meshes (56,995 vertices; 322,074 indices), dispatched
the Metal articulated operator, and bound each mesh to its named Core
inertial-body pose. It also rendered all 1,815 compiled muscle sites and
1,432 straight route-centreline segments in a now-retired diagnostic overlay.

The offline rest-frame fit is deliberately unchanged: its original 18
unambiguous segment anchors enumerate 24 proper signed axis maps and select
the identity axis map with positive scale `1.007736155369` after mm→m
conversion. Its equal-weight mesh-vertex-centroid to source-inertial-COM score
is `0.059372888 m` RMS (`0.123618266 m` maximum). The nine additions are
bilateral hip bones, fibulae, tali, patellae, and sternum body. They inherit
that fitted common frame instead of re-fitting to less meaningful centroids;
the fibulae attach visually to their ipsilateral tibial link because MyoSim has
no separate fibular segment. A mesh centroid and an inertial COM are not
homologous anatomical landmarks, so these remain common-frame plausibility
diagnostics—not surface-registration accuracy or a medical registration claim.

| View | PNG SHA-256 | Bone / site / route pixels | Inspection result |
| --- | --- | --- | --- |
| Front | `a611906f2f92ba02cc271e53cd095d14535d340a5d6a17bc4c942aed418dbc66` | `4,529 / 1,035 / 3,195` | bilateral hip bones, patellae, fibulae, and both leg/foot chains are visible with the complete path overlay |
| Oblique | `9eec3e986be00463e6f6482a4faa8427dba8952c01155606e9a34694d23ac04d` | `3,654 / 936 / 2,911` | sternum, shoulder/scapular depth, pelvis, and distal-leg additions are visible without a mirrored frame |
| Side | `4cd9519bcb1520cd368b50ebedf4540dcd19cd8cff5b645835d52fd78508139b` | `2,017 / 714 / 1,855` | sagittal skull–sternum–pelvis–patella–leg–foot sequence is continuous |
| Rear | `d8f1ae245b74b1f119289e1b3aace0eba5bc6645310473e5288f4ee929818eaf` | `4,797 / 1,003 / 3,558` | posterior bilateral limbs, fibulae, and sacral/pelvic connection are visible |

The native [transcript](media/myosim-native-bodyparts-major-bones-27/native-articulated-major-bones-27.transcript.txt),
visual-pack manifest, and `.mrvpack` accompany the four frames. This validates
the historical `Metal pose → articulated BodyParts3D bone instance → native
renderer` chain at the source default pose. The red overlay does **not**
validate tendon attachment and is not current showcase material.

## Retired bounded muscle-driven 27-bone route-overlay snapshot — 2026-08-27

Core `2aab522f92f44644c35bbde1a8ea3fd85356b027` captured these 640 × 640
frames on the Apple M4 Pro on `macmini`. The new native visual mode reads the
same verified `NHRIGID2`, `NHMYO1`, and `NHBONES1` inputs as the default-pose
binding, reconstructs all 416 source MuJoCo muscle definitions, projects their
source-default `0.5` excitation / `0.5` activation forces in Core FP64, and
advances one free-body step. Metal then computes the final 157-body pose; Core
bound the 27 BodyParts3D bone instances and complete site/route overlay to
that pose for rendering. There is no Python process in this capture, but its
straight-line overlay is retired for the same reason as the default-pose view.

The selected 1 ms step is deliberately a bounded visual sensitivity probe. It
applies all 90 source-default wraps and changes the active state relative to
the identically integrated passive state by maximum velocity
`71.4839058782` and configuration `0.0714839058782`. This large
co-activation response is exactly why the capture is **not** called a posture,
trajectory, gait, or physiological prediction: there is no controller,
contact, repeated integration, or stability qualification.

| View | PNG SHA-256 | Bone / site / route pixels | Inspection result |
| --- | --- | --- | --- |
| Front | `1c4f91fa09c4b1bba7482040a00e8f7d2a03d86941c079fde95222c454449c7a` | `4,525 / 1,044 / 3,237` | bilateral hip bones, patellae, fibulae, and limb chains remain coherent with the complete path overlay |
| Oblique | `359d06e9fc1f6107f0335e91e2d1c9360f42dc710b7813ec2cc77304ad29a405` | `3,697 / 931 / 2,943` | sternum, shoulder girdles, pelvic depth, and distal-leg additions remain visible after the force-driven state change |
| Side | `1d26bc3ef918e74c865a71b256d3581ae6565427ec28723ba32eba5211efd2eb` | `2,047 / 723 / 1,873` | sagittal skull–sternum–pelvis–patella–leg–foot sequence remains continuous |
| Rear | `d27152ce837f7eaec25cfd34f4422b6531d19fb2abea9423b16e9d1fe04c3c15` | `4,802 / 1,007 / 3,542` | posterior bilateral limbs, fibulae, and sacral/pelvic connection remain present |

The native [transcript](media/myosim-native-muscle-driven-major-bones-27/native-articulated-muscle-driven-major-bones-27.transcript.txt),
visual-pack manifest, and `.mrvpack` accompany the four frames. This closes the
bounded `complete 416-muscle force → articulated state step → Metal pose →
BodyParts3D bone renderer` evidence chain. It is not valid tendon-attachment
imagery and does not make the provisional bone transforms colliders; prove
deformable muscle bellies, skinning, organs, vessels, nerves, contact, a
sustained muscle-force rollout, or clinical anatomy separately.

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

1. Extend the inspected major-bone rest-frame binding to vertebrae, ribs,
   hands/digits, toes, and the remaining named skeleton meshes; matching
   anatomical labels remain insufficient evidence.
2. Add the deformable skin path after those reviewed skeletal attachments; do
   not replace it with rigid-bone parenting.
3. Resolve the Mortensen spine-to-`cervical_spine` rest registration and make
   an explicit MyoSim neck/head replacement decision before applying its 72
   cervical/hyoid muscle forces.
4. Replace the one-step sensitivity capture with a deterministic native
   free-body replay: persist `q`/`v` and the 416 activation states, advance
   source-defined activation dynamics from an explicit recorded control stream,
   emit a bounded frame sequence, and compare it with the matched passive
   replay. Until a controller and contact are validated, call that evidence a
   free-body response—not a posture or gait.
5. Move the complete MyoSim `J^T` force scatter and dense forward-dynamics
   update to a measured device-resident path, preserving CPU-vs-Metal replay
   parity before promoting the native capture to a live presentation sidecar.
6. Add registered anatomical colliders and calibrated contact before any
   standing or walking qualification.

This ordering keeps the Human more realistic by retaining source mechanics and
by refusing to turn a visually plausible mesh into an uncalibrated physical
body.
