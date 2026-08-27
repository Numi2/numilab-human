# Visual validation

## Native passive-FEM soleus check — 2026-08-27

The four-angle [passive soleus FEM specimen](VISUAL_PROGRESS.md#native-passive-fem-soleus-specimen--2026-08-27)
is the first native continuum slice in the Human visual path. It uses exact
`FJ1437` source-surface endpoints with the MyoSim-driven right tibia and
calcaneus as prescribed FEM end rings. The local Apple M4 completed eight
Matter steps for 12 nodes and nine tetrahedra after the all-416-path, 64-step
MyoSim drive, reporting `J_min = 0.900724887848`, two peak FGMRES iterations,
and nonzero bone/muscle/tendon/FEM coverage in front, oblique, side, and rear
views.

That validates the bounded native pose-to-FEM-to-render execution only. Its
Neo-Hookean material is explicitly uncalibrated, and the source tendon mesh is
still a separate visual surface rather than a force-transfer continuum. It
does not validate active muscle constitutive behavior, physiological tendon
attachment, full-body tissue, contact, gait, or clinical anatomy. Exact
device, inputs, image hashes, and limits are in the
[capture record](media/myosim-native-passive-fem-calf-2048/capture.transcript.txt).

## Selective right upper-limb source-actuator check — 2026-08-27

The 2048 px four-angle [upper-limb source-drive review](VISUAL_PROGRESS.md#focused-right-upper-limb-actuation--2026-08-27)
uses 42 BodyParts3D source bone meshes across the torso–scapula–humerus–forearm
chain and 20 exact source muscle surfaces. Rather than applying uniform
activation, it excites the ten pinned pectoralis/deltoid/coracobrachialis/elbow
flexor paths at `0.2`; the other 406 source actuators receive zero excitation.
All 416 source paths are still evaluated on the Apple M4 at each of 64
100 µs steps. The run published four non-empty bone/muscle views and a maximum
active/passive configuration difference of `0.0446275454086`.

This validates a selective source-actuator force path through the native
MyoSim/Core/Metal render sequence. It does not validate upper-limb contact,
an anatomical motion controller, stable movement, deformable tissue, or
clinical registration. Exact indices, counters, and image hashes are in the
[capture record](media/myosim-native-right-upper-limb-flexion-drive-2048/capture.transcript.txt).

## Current full-body muscle and tendon check — 2026-08-27

The [current four-angle full-body review](VISUAL_PROGRESS.md#current-full-body-source-muscle-and-tendon-review--2026-08-27)
keeps all 184 BodyParts3D bones and 150 named source surfaces visible after a
small, active all-416 route update. Its M4 execution has two Metal force
transactions, 416 active source records, 90 applied wraps, six final active
contacts among ten source-foot witnesses, and a `1.44013083483e-05`
configuration difference from the passive reference. The four views retain nonzero bone/muscle/tendon
coverage: 77,924/331,756/132, 72,578/273,205/770,
58,434/165,371/640, and 111,564/295,853/7,389 pixels.

Only the already source-triangle-locked distal boundary of each calcaneal
tendon can be visually closed to its named calcaneus. This avoids both the old
wrong two-body ownership and an invented muscle-to-tendon bridge, while making
the observed collagen-to-bone continuity readable at full-body scale. It does
not validate physical tendon attachment, a force-transfer law, deformable
tissue, contact, gait, or clinical registration. The exact native output is in
its [capture record](media/myosim-native-fullbody-nhtiss3-bone-collars-2048/capture.transcript.txt).

## Shared three-body Achilles check — 2026-08-27

The current 2048 px four-angle [Achilles review](VISUAL_PROGRESS.md#shared-three-body-achilles-review--2026-08-27)
replaces the invalid two-body tendon proxy. Exact BodyParts3D `FJ1405` now has
three explicit MyoSim/Core owners: `femur_r` (131), `tibia_r` (136), and
`calcn_r` (138); the mirrored surface has the corresponding left owners. The
distal source-triangle lock has 944 right and 943 left vertices within 3 mm of
the named calcaneal source surface, with 26 / 25 vertices in the 15 mm feather.
The rest capture retained nonzero tendon coverage in front, oblique, side, and
rear views: 1,715 / 4,329 / 9,407 / 21,648 pixels.

The linked selective contraction checks the same payload after one 100 µs
step. Only source muscles `348`, `349`, and `369` receive 0.5 activation;
all 416 paths remain device-evaluated on the Apple M4. The final configuration
differs from its passive reference by `0.000123820755509`, while the visible
anatomy stays in the small-displacement regime. Full commands, hashes,
coverage, and the explicit non-physical boundary are in the [capture record](media/myosim-native-three-body-achilles-2048/capture.transcript.txt).

This validates a source-body-weighted visual bind and a source force-path
inspection. It is not validation of deformable tendon mechanics, a
tendon-to-bone transfer law, contact, gait, or clinical registration.

## Selective source-actuator endpoint check — 2026-08-27

The four-angle 2048 px [selective posterior-calf route review](VISUAL_PROGRESS.md#selective-posterior-calf-source-actuator-route-review--2026-08-27)
isolates the actual source actuator route rather than inferring force
continuity from the visible collagen surface. The run excites only current
MyoSim actuator indices `348`, `349`, and `369`, yet executes every one of the
416 authored paths on the Apple M4 and subtracts the zero-activation baseline
before Core's bounded FP64 state/contact update. The final pose has nonzero
route segmentation in every frame: 2,886 / 9,757 / 3,998 / 16,711 pixels from
front through rear; the source route resolver reports two wraps and six
surface-projected endpoint cues.

Visual review confirms that the oblique, side, and rear views show the selected
routes ending on the articulated right calcaneal region rather than floating in
world space. These cyan lines are exact source route diagnostics evaluated at
the final pose, not a replacement tendon mesh or a physical attachment result.
The [capture record](media/myosim-native-selective-calf-route-attachment-2048/capture.transcript.txt)
contains the parameters, device counters, and exact images.

## Current source-surface-bound exterior check — 2026-08-27

The four-angle 2048 px [source-surface exterior review](VISUAL_PROGRESS.md#current-source-surface-bound-exterior-review--2026-08-27)
uses the exact BodyParts3D `FJ2810` skin mesh: 102,467 vertices and 203,382
triangles. The offline ABI-3 package selects four distinct bodies from 6,656
deterministic samples of exact registered source-bone surfaces across 86 Core
body bindings, and only preserves blends within a 12.5 mm local joint band.
It reconstructs the registered rest pose to `1.3383253895372864e-15 m`.

The native C++/Metal run on Apple M4 Pro executed all 416 authored source
paths at every one of 32 × 100 µs updates (64 Metal force transactions,
13,312 active records, and 2,866 wraps) before rendering its final 157-body
pose. The frames have 485,784 / 426,800 / 309,847 / 516,855 shell pixels from
front through rear, and visual inspection found one coherent exterior in all
four angles rather than the former oblique/lateral split silhouette. The
[capture record](media/myosim-native-skinned-fullbody-source-surface-2048/capture.transcript.txt)
contains output identities and devices.

This validates an articulated source-surface-local exterior binding under a
bounded muscle-force update. It does not validate physical skin deformation,
material calibration, collision/contact geometry, gait, or clinical
soft-tissue registration. The exposed tendon inspection remains the authority
for tendon-to-bone visual continuity.

## Retained driven tendon-junction check — 2026-08-27

The four-angle 2048 px [driven tendon-junction capture](VISUAL_PROGRESS.md#muscle-driven-tendon-junction-continuity--2026-08-27)
uses the exact 150-surface BodyParts3D muscle/tendon import after the same
32-step all-416-muscle Metal force update and Core FP64 source-foot-contact
fallback. At an open boundary whose source vertices are already locked to the
named distal body, the native renderer may close a visible raster seam with a
short source-proximity collar to that named bone surface only. It never
searches for or fabricates a muscle-to-tendon bridge. The rendered collar is
reported separately from source tendon pixels and follows the final articulated
pose.

This corrects the exposed anatomy presentation. It does not change MyoSim
spatial routes, attach a new force, weld a tendon to bone, define a material,
or prove physical tendon continuity, contact, gait, or clinical registration.
The exact inputs and image hashes are in its [capture record](media/myosim-native-supported-tendon-junction-2048/capture.transcript.txt).

## Presentation correction — 2026-08-27

The linked 640 × 640 native bone views use the former dense straight-line route
overlay. Retain them only as historical pose-to-mesh coverage evidence; they
are not current tendon imagery and cannot establish a tendon-to-BodyParts3D
surface attachment. The replacement renderer resolves source tangent contacts
and sampled wrap arcs only in an explicit focused diagnostic; its per-bone
attachment-site refinement is visual-only and remains outside physical
admission.

The earlier MyoSim source frames in [visual progress](VISUAL_PROGRESS.md) are
retained as Apache-2.0 provenance records but are retired from presentation.
They remain distinct from the BodyParts3D source-static evidence below.

## Current geometry-framed full-body anatomy check — 2026-08-27

The current broad visual reference is the four-angle 2048 px
[geometry-framed full-body inspection](VISUAL_PROGRESS.md#geometry-framed-full-body-source-anatomy--2026-08-27).
Its native camera framing is evaluated from the rendered source geometry, not
the narrower articulated COM envelope that cropped the earlier broad view.
The review has nonzero bone, muscle, and tendon segmentation in every angle,
including 5,502 tendon pixels in the rear frame. The camera is world-anchored
and targets the exact rendered-vertex centroid, avoiding the empty-space
oblique framing of the prior gallery. It covers 184 exact
BodyParts3D bone meshes and 150 source muscle/tendon surfaces at the
Metal-computed MyoSim pose on the local Apple M4.

This validates visibility and framing of the source anatomy at the captured
pose only. It is not skin realism, deformable muscle/tendon physics, contact,
stable support, a motion rollout, or medical registration.

## Current triangle-locked calcaneal-tendon check — 2026-08-27

The four-angle 2048 px [calcaneal-tendon inspection](VISUAL_PROGRESS.md#triangle-locked-calcaneal-tendon-inspection--2026-08-27)
uses the exact BodyParts3D tendon triangles and tests insertion vertices
against the exact named calcaneus triangle surface, not only against its
vertices. The 3 mm lock / 15 mm feather binds 944 + 26 right and 943 + 25
left vertices to their named calcanei. The native rear image retains 46,393
tendon pixels and visibly continues each Achilles mesh to its calcaneus.

This is a stronger presentation continuity check, not a tissue weld, tendon
force-transfer law, deformable tendon solve, dynamic contact result, or a
medical attachment validation.

## Current supported tendon attachment check — 2026-08-27

The current lower-leg reference is the four-angle 2048 px
[supported tendon attachment review](VISUAL_PROGRESS.md#supported-tendon-attachment-review--2026-08-27).
It runs the source-derived 5%-activation, 1 ms all-416-muscle state through a
bounded MyoSim foot-support contact solve before Metal poses and renders the
result. The exact BodyParts3D calcaneal-tendon triangles are insertion-locked
to the named calcaneus source mesh inside a 3 mm zone with a 12 mm feather;
the inspected active images retain tendon coverage from 1,654 to 22,281 pixels
per camera. The separate cyan route diagnostic resolves the MyoSim
gastrocnemius/soleus paths and makes their endpoint cues legible without using
a line as a tendon substitute.

The contact result is Core FP64 exact-cone, not GPU contact: the current Metal
full-dynamics contact bucket does not admit the 157-body connected tree. This
snapshot validates neither general collision nor a stable posture, gait,
deformable tendon, force transfer, or medical attachment.

## Current posterior-calf source-surface inspection — 2026-08-27

The current visual reference is the reviewed 2048 × 2048 four-angle
[posterior-calf source-surface inspection](VISUAL_PROGRESS.md#reviewed-native-posterior-calf-source-surface-inspection--2026-08-27).
It renders the exact BodyParts3D gastrocnemius, soleus, and calcaneal-tendon
triangles over a native two-body skeleton bind in isolated native reference
workspaces, so a later camera cannot reuse an earlier camera's observation.
All four views have nonzero bone, muscle-surface, and tendon-surface coverage.
The posterior view visibly continues the source tendon to the calcaneus.
Gastrocnemius is posed between femur/calcaneus, while soleus and the tendon are
posed between tibia/calcaneus; a source vertex has matching body-frame rest
evaluations and a bounded 0–1 per-vertex blend. This validates source-default
mesh visibility and kinematic crossing-surface continuity only; it does not
validate a physical attachment, MyoSim force transfer, or dynamic continuum
tendon mechanics.

## Current native full-skeleton visual check — 2026-08-27

The four default-pose and four 1 ms complete-muscle frames in
[visual progress](VISUAL_PROGRESS.md#reviewed-native-184-mesh-full-skeleton--2026-08-27)
were inspected from front, oblique, side, and rear views at 2048 × 2048. The
184 source meshes make the skull, atlas/axis, cervical/thoracic/lumbar spine, rib cage,
shoulders, hands/digits, pelvis, bilateral legs, and complete feet legible as
one skeleton. The earlier isolated ocular dot is gone: the renderer now uses
named cranial and mandibular meshes rather than `FJ1282`.

All eight frames have nonzero bone coverage. The default four demonstrate the
active Metal articulated-pose binding; the paired frames demonstrate the same
visual skeleton after the bounded complete-416-muscle free-body step. The
capture did not use the occupied Mac mini and is therefore local-Apple-M4
evidence. It does not validate skin or tendon deformation, physical tendon
attachments, contact, replay, standing, or gait.

## Retired native muscle-driven BodyParts3D major-bone overlay — 2026-08-27

The four M4 Pro frames in
[visual progress](VISUAL_PROGRESS.md#native-bounded-muscle-driven-27-bone-snapshot--2026-08-27)
re-run the current 27-mesh `NHBONES1` binding after one bounded full-body muscle
force step. Core `2aab522` projects all 416 source MyoSim muscles at the
source-default activation/excitation, advances an FP64 free-body state, and
passes only that resulting configuration to Metal for the final articulated
pose/render snapshot. The tracked transcript records 90 applied wraps and the
matched passive-state deltas, but the straight-line overlay cannot assess
surface attachment. This is a force-to-pose diagnostic, not a trajectory or a
biomechanical behavior claim: the 1 ms co-activation probe has no controller,
contact, recurrence, muscle-belly/skin deformation, or stability validation.

## Retired native articulated BodyParts3D major-bone overlay — 2026-08-27

The four inspected frames in
[visual progress](VISUAL_PROGRESS.md#native-bodyparts3d-27-major-bone-binding--2026-08-27)
are historical pose-bound BodyParts3D geometry evidence. An offline import
uses an 18-mesh unambiguous similarity-fit set, then writes 27 exact source
bone meshes and their link-local uniform-scale transforms into `NHBONES1`.
The native Core capture binds each record to its Metal-computed MyoSim
inertial-body pose.

This establishes a shared default-frame candidate and an executable
`articulated pose → BodyParts3D bone instance → renderer` chain. It is stronger
than the static skin preview below, but remains deliberately narrower than a
physical registration: the centroid/COM score is a common-frame diagnostic,
not a surface-landmark residual; fibulae use the ipsilateral tibial link because
the active MyoSim source has no separate fibular body; collision/contact, skin
weights, unregistered small bones, soft-tissue deformation, and motion-replay
qualification are not included.

## Retired native BodyParts3D skin source snapshot — 2026-08-27

The exact CC-BY-4.0 `FJ2810` full-skin OBJ was converted from source millimetres
to metres into a GLB, cooked into a Core visual pack, and rendered through
`metalrobo_bodyparts3d_visual_probe` on the local Apple M4 at Core `86790f3`.
The 512 × 512 output is retained as cooking/visibility evidence, but is retired
from presentation while high-resolution source anatomy references are reviewed.
The cooked mesh contains 102,467 vertices and 203,382 triangles. Each camera
had nonzero source coverage: 13,045 anterior, 8,455 oblique, and 13,345
posterior pixels.

| View | PNG SHA-256 | Inspection result |
| --- | --- | --- |
| Anterior | `4dcff97fe2bbf275cfe3afa2b8e2bc23a0e2eaa616925cf5495f7b25e30f8bcd` | upright face, torso, bilateral arms/hands, legs, and feet visible |
| Oblique | `d6823e87efb7c444a9c1072b6dd539f01cd4580a0d83b89e695f68fb72c68691` | continuous head–shoulder–torso–pelvis–leg silhouette visible |
| Posterior | `f782071f9713112e4244821759e8e9c9e8401aa785ea818160d2ea125791186a` | rear contour, arms/hands, legs, and feet visible |

This confirms the complete source skin survives native Core cooking and
multi-angle rendering. It does **not** establish a MyoSim-body transform,
skinning weights, collision, deformable shell mechanics, or a live articulated
surface; those require registration evidence rather than a plausible overlay.

## BodyParts3D full-skin preview — 2026-08-26

The visual preview begins with the exact `FJ2810` skin OBJ from
`isa_BP3D_4.0_obj_99.zip`. It is converted only from source millimetres to
preview metres; it is not registered to a Rajagopal body frame or the Core
FunctionBased state. This deliberately keeps visual-source validation separate
from the device-qualified skeleton and Millard-actuation path.

| Item | Value |
| --- | --- |
| BodyParts3D archive SHA-256 | `40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e` |
| Source OBJ member / SHA-256 | `isa_BP3D_4.0_obj_99/FJ2810.obj` / `682f402206f15592acdeaae8ffb6b34c3e5c3267fa4685e63d2e4920ef2a80e0` |
| Source surface | 102,467 vertices; 203,382 triangles |
| Core revision | `14c64f303adb713f3a011546908688adb5848c61` (`origin/coupled`) |
| Cooked `.mrvpack` SHA-256 | `5b78d852357ea3cbfe44a9d0d55fb9a68251b8971d3bdfa4a0806750267063b9` |
| Visual probe SHA-256 | `7cb286f927a4d31d323d90a87d748fb4e3678945e0fd9838c647bb25b04f2b0d` |
| Core library SHA-256 | `08d548091af84d460f1c326cf3b7cc6b67fa89b2fbd266ed9b585d1a69a0d59d` |
| Metal library SHA-256 | `8351863cbbf5ce523956d9b49484ae39c315f9b8bdec54285544bcffaff71922` |
| GPU | Apple M4 Pro on `macmini` |
| Render profile | `sensor_reference`, 512 × 512, one static environment |

The checked final views are an anterior-looking `axis_negative_y` view, an
upright side-oblique view, and a posterior-looking `axis_positive_y` view.
Their BodyParts3D source-pixel counts were respectively 13,045, 8,455, and
13,345. The final PPM SHA-256 values were, in that order,
`c95035994447b26cf18dc03a81e9a9ed519e7d63f787effffd7acf4c0351337a`,
`5e0120ee16039f09e5581f8d65a37235febbc75e495705665e70850b3ac38011`, and
`6e0c0fb5fe9ed8d7161c0e47cdaa80cc8baf556fcfdda2e74600b6e32a659bcd`.
The device log SHA-256 is
`21a5d9f344d5384c9ca1c4aba80eba0b6c8cce4a583683ad7f6beee4f2d37c1e`.

The front and rear contours, limbs, hands, head, and feet were visible and
upright in the inspected frames. The first camera-basis implementation exposed
roll instability in oblique and opposite-axis views; the final render uses a
world-up-preserving basis and was rechecked at all three angles.

These frames are retained outside Git because they are derived from a
third-party BodyParts3D geometry source. They validate source-surface cooking,
camera framing, and renderer visibility only. They do **not** validate anatomy
registration, skinned deformation, joint motion, muscles, collision, contact,
organ/vessel mechanics, tissue material parameters, or a full Human RobotPack.

## Source-static anatomy-layer previews — 2026-08-26

`numi human visual-layers` selected the largest exact source mesh for each
requested layer, then each GLB was cooked and rendered by
`metalrobo_bodyparts3d_visual_probe` on the Apple M4 Pro. All three stable
inspection cameras contained source pixels for every layer.

| Layer | Source mesh | Vertices / triangles | Front / oblique / rear pixels |
| --- | --- | --- | --- |
| Skin | `FJ2810` | 102,467 / 203,382 | 13,045 / 8,455 / 13,345 |
| Bone | `FJ1368` | 20,582 / 39,524 | 14 / 9 / 14 |
| Muscle | `FJ1451` | 85,775 / 98,928 | 507 / 488 / 538 |
| Vessel | `FJ2145` | 15,163 / 30,036 | 58 / 51 / 67 |
| Nerve | `FJ1806` | 15,757 / 26,512 | 110 / 155 / 120 |

This used the isolated Core `14c64f3` visual worktree and its M4 Pro probe.
The packs and PPM captures are retained outside Git at
`/Users/n/numilab-human-layer-renders-2d570d5-v2`; they are third-party
geometry-derived evidence. This validation establishes only source-static
layer visibility from three angles. It does not establish an anatomical
attachment transform, skinned motion, collision/contact, tissue mechanics, or
a walking Human.
