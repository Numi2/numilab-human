# NumiLab Human v1

An Apple-native, muscle-driven Human foundation for Numi Lab. The active
full-body mechanical source is **MyoSim `myofullbody`** (416 authored
muscle-tendon actuators); the Core owns the articulated state, muscle route
evaluation, force scatter, and forward dynamics. There is no Python process in
the native Human execution path.

| Role | Source | Status |
| --- | --- | --- |
| Active full-body mechanics | MyoSim `myofullbody` | 103 source bodies, 416 muscles, native Core reference |
| Cervical/hyoid mechanics | Mortensen 2018 | complete 72-muscle OpenSim 3 source IR; merge registration remains explicit |
| Anatomy/visual layers | BodyParts3D 4.0 | named geometry/hierarchy; 184 source bone meshes are pose-bound for native visual inspection |
| Detailed calf visual supplement | Z-Anatomy | four right-calf surfaces only; CC-BY-SA geometry, BodyParts3D/MyoSim body bindings retained |
| Comparative lower-body mechanics | RajagopalLaiUhlrich2023 | retained source-faithful bounded Metal path |
| Comparative upper extremities | MoBL-ARMS | authenticated bimanual import or pinned public unimanual 4.1 source variant |

The importer preserves upstream records locally. The tracked MyoSim,
BodyParts3D, and explicitly marked Z-Anatomy validation media are attributed
derivatives; all other raw or derived source artifacts remain local. See
[third-party notices](THIRD_PARTY_NOTICES.md).

## Visual progress

The lead visual pairs exposed source anatomy—where muscles and tendons can be
inspected directly against named bones—with a source-surface-bound exterior
that remains coherent under the bounded all-muscle probe. See
[visual progress](Docs/VISUAL_PROGRESS.md) for the exact evidence boundary.

### Detailed muscle-driven right-calf anatomy

<p align="center">
  <img src="Docs/media/myosim-native-zanatomy-calf-2048/calcaneal-insertion/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-zanatomy-calf-supplement-muscle-driven-selected-actuators-focus-body-138-oblique.png" width="49%" alt="Detailed right calcaneal tendon insertion, oblique" />
  <img src="Docs/media/myosim-native-zanatomy-calf-2048/calcaneal-insertion/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-zanatomy-calf-supplement-muscle-driven-selected-actuators-focus-body-138-rear.png" width="49%" alt="Detailed right calcaneal tendon insertion, rear" />
</p>

The visual-only right-calf slice uses the CC-BY-SA 4.0 Z-Anatomy lateral and
medial gastrocnemius, soleus, and calcaneal-tendon meshes. Its named calcaneus
is registered to BodyParts3D `FJ3360`; its four tissues retain the existing
MyoSim femur/tibia/calcaneus bindings, including the Achilles three-body
ownership. The distal tendon does not merely follow the calcaneus by a blend:
its 96 source-triangle-locked vertices and 155-vertex feather band are
registered directly to named `FJ3360` calcaneus triangles (0.35 mm exterior
offset). The 2K capture runs the three named calf actuators at `0.5` for one
100 µs step while Metal evaluates all 416 source paths, so it has a measured
small pose delta (`0.000123820755509`) rather than an uncontrolled pose drift.
This makes the muscle-to-tendon-to-calcaneus geometry inspectable, but it is
still an anatomical visual plate—not photorealistic skin, a tendon continuum,
or a physical attachment certificate. The [capture record](Docs/media/myosim-native-zanatomy-calf-2048/capture.transcript.txt) records the source and output identities.

### Muscle-driven source-surface exterior

<p align="center">
  <img src="Docs/media/myosim-native-skinned-fullbody-source-surface-2048/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell-muscle-driven-source-support-contact-front.png" width="49%" alt="Source-surface-bound muscle-driven Human exterior, front" />
  <img src="Docs/media/myosim-native-skinned-fullbody-source-surface-2048/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell-muscle-driven-source-support-contact-oblique.png" width="49%" alt="Source-surface-bound muscle-driven Human exterior, oblique" />
</p>

This Apple-M4-Pro 2K exterior review uses the exact 102,467-vertex BodyParts3D
skin mesh. Each vertex selects from 6,656 sampled exact source-bone surface
points across 86 registered articulated bodies, with blending limited to a
12.5 mm local joint band. All 416 authored MyoSim paths ran on Metal for 32 ×
100 µs bounded updates before the native render. It is a coherent articulated
source exterior, not a textured avatar, deformable skin/tissue solve,
collision shell, gait result, or clinical registration.

### Muscle-driven full-body anatomy

<p align="center">
  <img src="Docs/media/myosim-native-fullbody-nhtiss3-bone-collars-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-front.png" width="49%" alt="All-muscle BodyParts3D Human, front" />
  <img src="Docs/media/myosim-native-fullbody-nhtiss3-bone-collars-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-rear.png" width="49%" alt="All-muscle BodyParts3D Human, rear" />
</p>

This separate Apple-M4 2K anatomy pass is the active full-body mechanical
view: all 416 authored MyoSim paths ran on Metal before one bounded 100 µs
state update. It renders 184 BodyParts3D bone meshes and 150 named
muscle/tendon surfaces. The two Achilles meshes use their corrected three-body
femur/tibia/calcaneus ownership, and their source-locked distal boundary may
receive a short collar only onto the named calcaneus—not an inferred
muscle-to-tendon bridge. It is an anatomy/force-path inspection, not a
deformable tissue or tendon result.

### Shared-tendon, source-body attachment review

<p align="center">
  <img src="Docs/media/myosim-native-three-body-achilles-2048/rest/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-front.png" width="24%" alt="Three-body Achilles binding, front" />
  <img src="Docs/media/myosim-native-three-body-achilles-2048/rest/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-oblique.png" width="24%" alt="Three-body Achilles binding, oblique" />
  <img src="Docs/media/myosim-native-three-body-achilles-2048/rest/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-side.png" width="24%" alt="Three-body Achilles binding, side" />
  <img src="Docs/media/myosim-native-three-body-achilles-2048/rest/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-rear.png" width="24%" alt="Three-body Achilles binding, rear" />
</p>

The previous two-body Achilles presentation was wrong: it reduced the two
gastrocnemius femoral origins and soleus tibial origin to one tibia–calcaneus
blend. The `NHTISS3` source-surface payload instead binds each exact
BodyParts3D Achilles mesh to femur, tibia, and calcaneus. Its proximal weights
are inherited from the nearest named gastrocnemius/soleus source surface, and
944 right / 943 left distal source vertices are locked to the exact calcaneal
triangle surface. The four static images are a clean anatomy inspection; the
[capture record](Docs/media/myosim-native-three-body-achilles-2048/capture.transcript.txt)
also records the selective native muscle-driven check. This remains a
kinematic visual surface, not a tendon continuum, weld, force-transfer law,
contact result, gait result, or clinical attachment certificate.

### Selective muscle-to-bone route review

<p align="center">
  <img src="Docs/media/myosim-native-three-body-achilles-2048/selective-drive/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-surface-projected-sites-focus-body-136-front.png" width="24%" alt="Selective calf route review, front" />
  <img src="Docs/media/myosim-native-three-body-achilles-2048/selective-drive/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-surface-projected-sites-focus-body-136-oblique.png" width="24%" alt="Selective calf route review, oblique" />
  <img src="Docs/media/myosim-native-three-body-achilles-2048/selective-drive/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-surface-projected-sites-focus-body-136-side.png" width="24%" alt="Selective calf route review, side" />
  <img src="Docs/media/myosim-native-three-body-achilles-2048/selective-drive/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-surface-projected-sites-focus-body-136-rear.png" width="24%" alt="Selective calf route review, rear" />
</p>

This is the mechanical counterpart to the collagen-surface review. Only the
current pinned MyoSim gastrocnemius lateralis/medialis and soleus actuators
(348/349/369) receive `0.5` excitation for one 100 µs step; Metal still
evaluates all 416 authored paths before the bounded dynamics step. Cyan is the resolved source
spatial muscle–tendon route with endpoint cues projected to the articulated
BodyParts3D bones, so its rear and oblique views make the actual calcaneal
endpoints inspectable. The [capture record](Docs/media/myosim-native-three-body-achilles-2048/capture.transcript.txt)
contains the device counters and frame hashes. This is a force-path diagnostic,
not a tendon mesh, continuum, force-transfer certificate, gait, or clinical
attachment claim.

### Retired two-body tendon imagery

The older isolated tibia–calcaneus tendon images are retained only as
reproducibility artifacts. They are not presented as current anatomy because
they omit the gastrocnemius femoral ownership now represented in the shared
three-body review above.

### Source skin provenance

The exact 102,467-vertex, 203,382-triangle BodyParts3D `FJ2810` shell remains
the exterior source. Its current native payload uses source-bone surface
samples—not broad bone-box proximity—to choose four candidate bodies per
vertex, then restricts nonzero blends to the local source joint band. It is
not a deformable-shell mechanics result or a human-quality textured avatar.

### Selective upper-limb source-actuator drive

<p align="center">
  <img src="Docs/media/myosim-native-right-upper-limb-flexion-drive-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-focus-body-41-front.png" width="24%" alt="Right upper-limb source drive, front" />
  <img src="Docs/media/myosim-native-right-upper-limb-flexion-drive-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-focus-body-41-oblique.png" width="24%" alt="Right upper-limb source drive, oblique" />
  <img src="Docs/media/myosim-native-right-upper-limb-flexion-drive-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-focus-body-41-side.png" width="24%" alt="Right upper-limb source drive, side" />
  <img src="Docs/media/myosim-native-right-upper-limb-flexion-drive-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-focus-body-41-rear.png" width="24%" alt="Right upper-limb source drive, rear" />
</p>

This torso–scapula–humerus–forearm view uses 42 source bone meshes on
six articulated bodies and 20 exact BodyParts3D muscle surfaces. It excites
only ten named MyoSim sources: three pectoralis-major slips, anterior and
acromial deltoid, coracobrachialis, both biceps heads, brachialis, and
brachioradialis. The remaining 406 are set to zero excitation, but Metal still
evaluates all 416 authored routes at every one of the 64 × 100 µs updates.
The active/passive configuration difference is `0.0446275454086`; the
[capture record](Docs/media/myosim-native-right-upper-limb-flexion-drive-2048/capture.transcript.txt)
contains exact muscle indices, device counters, and frame hashes. This is a
bounded Apple-M4 muscle-force/free-dynamics inspection—not contact, a
controller, deformable soft tissue, tendon continuum, stable movement, or
clinical-registration evidence.

### Reviewed full-skeleton native inspection

<p align="center">
  <img src="Docs/media/myosim-native-full-skeleton-184-2048/default/myosim-fullbody-articulated-bodyparts-bones-front.png" width="32%" alt="Full BodyParts3D visual skeleton, front" />
  <img src="Docs/media/myosim-native-full-skeleton-184-2048/default/myosim-fullbody-articulated-bodyparts-bones-oblique.png" width="32%" alt="Full BodyParts3D visual skeleton, oblique" />
  <img src="Docs/media/myosim-native-full-skeleton-184-2048/default/myosim-fullbody-articulated-bodyparts-bones-rear.png" width="32%" alt="Full BodyParts3D visual skeleton, rear" />
</p>

This four-angle 2048 × 2048 native inspection binds 184 exact BodyParts3D
bone meshes—cranial bones, vertebrae, ribs, hands, feet, and the major
limbs—to 86 active MyoSim rigid parents and the Metal-computed 157-body pose.
A separate four-angle 1 ms snapshot uses all 416 source muscle-tendon forces
before the final native render. These are visual-skeleton and bounded
force-to-pose evidence, respectively; neither makes skin, tendons, organs,
contact, gait, or clinical-registration claims. See [visual progress](Docs/VISUAL_PROGRESS.md#reviewed-native-184-mesh-full-skeleton--2026-08-27).

### Native anatomy presentation reset

The previous native red-line galleries are retained as diagnostic artifacts but
are no longer presented as Human anatomy. They drew every source route as a
straight segment between sites and wrap centres, so a line could cut through a
wrap object or visibly miss a BodyParts3D surface. That is not an acceptable
tendon view.

The current native renderer resolves the source tangent contacts and samples
the selected sphere/cylinder wrap arcs at the rendered pose. Its default
anatomy view hides route lines; its focused inspection mode renders only a
chosen muscle set around one MyoSim link at 1024 × 1024, with source-site
endpoints visually projected to the nearest matching BodyParts3D bone triangle.
That projection never alters the force solver. The offline BodyParts3D
registration also has a visual-only per-bone attachment-site refinement. These
are inferred correspondences—not tendon-surface geometry or a medical
attachment certificate—so refined captures are reviewed before replacing the
gallery. See [visual progress](Docs/VISUAL_PROGRESS.md) for the exact boundary.

## Native full-body execution

After a local artifact has been acquired and compiled, the production-facing
reference needs only the Apple-native Numi Core executable:

```sh
# No Python process is started by this command. `--metal` additionally
# executes full-body pose/Jacobians plus all MyoSim route and static-force
# evaluations on the Apple GPU.
numi human myosim-native-probe Build/myosim-fullbody --metal

# Render the same compiled payload through the native articulated-marker view.
# This command starts no Python process.
numi human myosim-native-visuals \
  Build/myosim-fullbody \
  Docs/media/myosim-native-articulated

# Offline source registration/package preparation, followed by a native
# C++/Metal visual-skeleton capture.  The final command starts no Python
# process. The attachment refinement is optional and reserved for focused
# source-site inspection; the broad visual skeleton uses the common-frame
# candidate directly.
numi human myosim-bodyparts-registration \
  --sources Sources --artifact Build/myosim-fullbody \
  --output Build/myosim-fullbody/bodyparts3d-major-bone-registration.candidate.json
numi human myosim-bodyparts-bone-payload \
  --sources Sources \
  --registration Build/myosim-fullbody/bodyparts3d-major-bone-registration.candidate.json \
  --output Build/bodyparts3d-myosim-major-bones
numi human myosim-native-bone-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Docs/media/myosim-native-full-skeleton-184-2048/default \
  --dimension 2048

# Native bounded force-to-pose sensitivity capture (no Python process).  The
# 1 ms limit is an inspection step, not an uncontrolled rollout duration.
numi human myosim-native-muscle-bone-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Docs/media/myosim-native-full-skeleton-184-2048/muscle-driven \
  --muscle-step-seconds 0.001 --dimension 2048

# Inspect a named source subset by its manifest actuator indices. This native
# diagnostic resolves tangent contacts and wrap arcs; it is not a rendered
# muscle belly or a surface-attachment certificate. 348/349/369/371 are the
# right gastrocnemius lateralis/medialis, soleus, and tibialis anterior in the
# current pinned MyoSim manifest; body 136 is its right tibia.
numi human myosim-native-route-inspection \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/route-inspection-right-lower-leg \
  136 348 349 369 371

# To inspect a real selective contraction, add a bounded step. The command
# excites the listed source actuators while still evaluating all 416 MyoSim
# paths on Metal, retains the source-foot support fallback when present, and
# draws the resolved route plus its endpoint cues on the linked bones. It does
# not turn the route diagnostic into a deformable tendon surface.
numi human myosim-native-route-inspection \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/route-inspection-right-calf-selective-contraction \
  136 348 349 369 \
  --muscle-step-seconds 0.0001 --muscle-step-count 64 \
  --muscle-activation 0.5 --dimension 2048

# Prepare exact BodyParts3D posterior-calf muscle/tendon surfaces in the same
# source-default frame, then render them over the focused native skeleton.
# The current package requires the complete v2 184-mesh registration and uses
# a kinematic two-body bind, not deformable tissue mechanics.
numi human myosim-bodyparts-right-posterior-chain-payload \
  --sources Sources \
  --registration Build/myosim-fullbody/bodyparts3d-major-bone-registration.candidate.json \
  --output Build/bodyparts3d-myosim-right-posterior-chain
numi human myosim-native-soft-tissue-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-right-posterior-chain/bodyparts3d-myosim-right-posterior-chain.nhtissue \
  Docs/media/myosim-native-posterior-chain-2048/default \
  136 --dimension 2048

# The same exact surfaces after a bounded all-416-muscle *incremental*
# activation step. Zero-activation source pre-stress is subtracted before the
# force step; this remains a coupling check, not stable/contact-qualified motion.
numi human myosim-native-muscle-soft-tissue-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-right-posterior-chain/bodyparts3d-myosim-right-posterior-chain.nhtissue \
  Build/myosim-native-posterior-chain-muscle-stress \
  136 --muscle-step-seconds 0.001 --muscle-activation 0.05 --dimension 2048

# The source-authored foot primitives provide a bounded support-contact
# snapshot before the final Metal pose/render. This native command reports
# whether contact was admitted to Metal; the full 157-body tree currently uses
# the Core FP64 exact-cone fallback and never claims GPU contact when rejected.
numi human myosim-native-supported-muscle-soft-tissue-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-right-posterior-chain/bodyparts3d-myosim-right-posterior-chain.nhtissue \
  Docs/media/myosim-native-supported-posterior-chain-attachment-2048 \
  136 --muscle-step-seconds 0.001 --muscle-activation 0.05 --dimension 2048

# Package the audited full-body muscle-surface map. The offline importer
# verifies every ordinary surface against its exact BodyParts3D FJ mesh and
# named MyoSim route endpoints; the native capture remains Python-free.
numi human myosim-bodyparts-fullbody-muscle-surface-payload \
  --sources Sources \
  --registration Build/myosim-fullbody/bodyparts3d-major-bone-registration.candidate.json \
  --artifact Build/myosim-fullbody \
  --output Build/bodyparts3d-myosim-fullbody-muscle-surfaces
numi human myosim-native-fullbody-soft-tissue-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-fullbody-muscle-surfaces/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue \
  Docs/media/myosim-native-fullbody-geometry-framed-2048 \
  --dimension 2048

# Optional detailed right-calf geometry. Blender is used only for this
# offline, CC-BY-SA source export; neither native inspection command starts
# Blender or Python. The payload retains MyoSim/BodyParts3D body bindings and
# projects only the already source-triangle-locked tendon insertion band onto
# the named BodyParts3D calcaneus surface.
/opt/homebrew/bin/blender --background /path/to/Startup.blend \
  --python tools/export_zanatomy_calf.py -- Build/zanatomy-calf-export.json
numi human zanatomy-calf-visual-supplement-payload \
  --sources Sources \
  --registration Build/myosim-fullbody/bodyparts3d-major-bone-registration.candidate.json \
  --base-payload Build/bodyparts3d-myosim-fullbody-muscle-surfaces/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue \
  --zanatomy-export Build/zanatomy-calf-export.json \
  --output Build/zanatomy-calf-myosim-tissues
numi human myosim-native-zanatomy-calf-inspection \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/zanatomy-calf-myosim-tissues/zanatomy-calf-myosim-tissues.nhtissue \
  Docs/media/myosim-native-zanatomy-calf-2048 --dimension 2048
numi human myosim-native-zanatomy-calcaneal-insertion \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/zanatomy-calf-myosim-tissues/zanatomy-calf-myosim-tissues.nhtissue \
  Docs/media/myosim-native-zanatomy-calf-2048/calcaneal-insertion --dimension 2048

# Default whole-body, ground-supported muscle-force presentation. This starts
# no Python process: all 416 MyoSim routes and activation sidecars run on
# Metal; the present 157-body contact island uses the explicitly reported
# Core-FP64 contact fallback before the final native Metal render.
numi human myosim-native-supported-fullbody-muscle-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-fullbody-muscle-surfaces/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue \
  Docs/media/myosim-native-fullbody-supported-metal-force-2048 \
  --muscle-step-seconds 0.0001 --muscle-step-count 32 \
  --muscle-activation 0.05 --dimension 2048

# Build the exact exterior BodyParts3D shell once, then run the separate
# Python-free native muscle-driven presentation.  The offline import derives
# four registered visual influences per vertex and verifies its rest-pose
# reconstruction; it does not claim a physical skin material or collision.
numi human myosim-bodyparts-skinned-shell-payload \
  --sources Sources \
  --registration Build/myosim-fullbody/bodyparts3d-major-bone-registration.candidate.json \
  --output Build/bodyparts3d-myosim-skinned-shell
numi human myosim-native-supported-skinned-fullbody-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-skinned-shell/bodyparts3d-myosim-skinned-shell.nhskin \
  Docs/media/myosim-native-skinned-fullbody-metal-force-2048 \
  --muscle-step-seconds 0.0001 --muscle-step-count 32 \
  --muscle-activation 0.05 --dimension 2048

# Four-angle calcaneal insertion review. This consumes a single matched pair
# of source-prepared visual payloads; the native executable rejects a mixed
# visual-registration pair before rendering. No Python process is started.
numi human myosim-native-calcaneal-tendon-detail \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-fullbody-muscle-surfaces/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue \
  Docs/media/myosim-native-calcaneal-tendon-detail-2048 \
  --dimension 2048

# Run the named right shoulder/elbow flexion set. Its ten source actuators are
# explicit; every one of the 416 MyoSim routes is still evaluated natively.
# This command starts no Python runtime.
numi human myosim-native-right-upper-limb-flexion-drive \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-fullbody-muscle-surfaces/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue \
  Docs/media/myosim-native-right-upper-limb-flexion-drive-2048 \
  --dimension 2048

# Export a separate exact BodyParts3D rest-frame reference for the right lower
# leg. It contains source bone, muscle, and calcaneal-tendon surfaces and is
# intentionally not claimed as an articulated or physical attachment. Render
# it with `metalrobo_bodyparts3d_visual_probe --dimension 1024
# --focus-lower-third` for a useful muscle/tendon inspection scale.
numi human right-lower-leg-anatomy-preview \
  --sources Sources \
  --output Build/bodyparts3d-right-lower-leg-anatomy

# A more focused posterior-chain source inspection. It preserves authored OBJ
# normals, excludes anatomy that would hide the tendon, and is static only.
numi human right-calcaneal-tendon-continuity-preview \
  --sources Sources \
  --output Build/bodyparts3d-right-calcaneal-tendon-continuity
```

This loads the `NHRIGID2` and `NHMYO1` payloads directly into Core, validates
the 128-DoF floating articulated tree, evaluates every muscle route, applies
the resulting generalized force, and executes forward dynamics. With `--metal`,
the same command buffer also evaluates all 416 MuJoCo-source spatial routes,
their default-state static actuator forces, and the complete source
`J^T` generalized-force projection on the Apple GPU, without restaging poses
through the CPU. The same native transaction can then advance every valid
MyoSim activation sidecar by one explicit `100 µs` Metal step; the probe uses
non-equilibrium source states and requires the reusable command-buffer path to
publish the same next state as the one-shot path. On Apple M4, the device’s
maximum per-muscle force-vector
error is `0.00471758869298`; the deterministic all-416 reduction differs by
`0.00642352090836` from the FP64 source reference. The activation result is
currently returned with the operator result; it is not yet a persistent
device-only full-body rollout. Dense 128-DoF device mass dynamics, contact,
skin/organ solvers, and clinical qualification remain separate work.

The bounded `--muscle-step-*` visual path now consumes that same retained
Metal force transaction for both the active and zero-activation baseline at
each step. Core FP64 still receives only the two returned 128-DoF force
vectors to advance the current full-body state; the final pose and all camera
renders remain native Metal. This removes the former host loop that projected
416 individual muscle paths during a visual update, without overstating the
remaining dynamics/contact boundary.

The same native reference also compares one unconstrained 1 µs FP64 state step
with the complete 416-muscle generalized force against the identical passive
state. The active route force changes velocity by `0.0714839058782` and
configuration by `7.14839058782e-08`; this is a bounded state-coupling smoke
test, not stable movement, contact, gait, or physiological calibration.

## Offline source import

MyoSim's own upstream composition API is Python. It is used *only* to acquire
and serialize source values into immutable native payloads; it is not part of
the Numi runtime, control loop, or physics execution. Keep it isolated in the
source environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -e .

# Fetch the pinned Apache-2.0 MyoSim and MIT Mortensen source checkouts.
numi human myosim-fetch --output Sources

# Offline source compilation only; generated payloads then run natively above.
numi human myosim-build \
  --sources Sources \
  --python .venv-myosim/bin/python \
  --output Build/myosim-fullbody

numi human mortensen-neck \
  --sources Sources \
  --output Build/mortensen-neck/mortensen-neck-source.ir.json

# `numi human` is this repository's workspace capability. It fetches only
# BodyParts3D 4.0 and the pinned Rajagopal model; it never tries
# to bypass the SimTK login required for MoBL-ARMS.
numi human fetch --output Sources

# Download the original bimanual MoBL-ARMS archive while signed into SimTK,
# then build a local Human v1 manifest.
numi human build \
  --sources Sources \
  --upper-archive /path/to/MobL_ARMS_OpenSim3_bimanual_model.zip \
  --accept-upper-noncommercial-terms \
  --output Build/human-v1

# For non-commercial research, the pinned public unimanual MoBL-ARMS 4.1
# source variant can be fetched and imported explicitly. It does not replace
# the original authenticated bimanual release above.
numi human fetch \
  --output Sources \
  --include-public-mobl-41 \
  --accept-upper-noncommercial-terms
numi human build \
  --sources Sources \
  --upper-public-mobl-41 \
  --accept-upper-noncommercial-terms \
  --output Build/human-v1-public-unimanual

# Audit the active MyoSim full-body route separately from the legacy stitched
# BodyParts3D + Rajagopal + authenticated MoBL-ARMS manifest.
numi human audit \
  --sources Sources \
  --runtime-root /path/to/MetalRobo \
  --output Build/human-v1-gates.json

# Fingerprint every separate BodyParts3D OBJ and conservatively preflight its
# surface topology. This does not repair or convert a source mesh.
numi human geometry-audit --sources Sources --output Build/bodyparts3d-topology.json

# Export the exact BodyParts3D full-skin OBJ as a source-static visual preview.
# This is intentionally unregistered to OpenSim frames and has no physics semantics.
numi human visual-preview --sources Sources --output Build/bodyparts3d-skin-preview

# Emit the source-derived mobile pelvis and 80-muscle learned-walking contract.
# It records the bounded synthetic contact-response path but leaves anatomical
# collider registration/contact calibration as gated work.
numi human walking-contract --sources Sources --output Build/rajagopal-walking-contract.json

# Build and run the immediate goal: a mobile, muscle-driven lower body on
# flat ground. The four simple foot pads are temporary engineering scaffolding,
# not anatomical BodyParts3D foot geometry.
numi human pilot --sources Sources --output Build/lower-body-pilot --smoke

# Produce review-only lower-body anatomy attachment and foot-collider work items.
numi human attachment-worklist --sources Sources --output Build/lower-body-attachments.json

# Create the provenance-pinned, fail-closed hand-off for the four source foot
# bodies. A reviewer must add transforms and collider/calibration receipts;
# this command never infers them from names.
numi human foot-registration-template --sources Sources --output Build/foot-registration-template.json

# Derive hash-pinned, source-local enclosing-box candidates for the same foot
# meshes. These are not OpenSim-frame colliders until the reviewed transform
# and contact-calibration receipts are supplied.
numi human foot-collider-preflight --sources Sources --output Build/foot-collider-preflight.json

# Combine the exact source identities and source-local boxes into one blank,
# provenance-pinned reviewer receipt. It still cannot supply transforms or
# calibrated contact values on the reviewer's behalf.
numi human foot-registration-receipt-template --sources Sources --output Build/foot-registration-receipt-template.json

# After a reviewer completes the receipt, verify its source hashes, rigid
# transform math, three-view evidence, and contact-field completeness. This
# does not turn the review into a physics or walking qualification.
numi human foot-registration-receipt-check --sources Sources --receipt Build/reviewed-foot-receipt.json --output Build/reviewed-foot-receipt-validation.json

# Export exact source-static skin, bone, muscle, vessel, and nerve layer previews.
numi human visual-layers --sources Sources --output Build/bodyparts3d-layers

The five source-static anatomy layers have passed three-angle Apple M4 Pro
inspection; see [visual validation](Docs/VISUAL_VALIDATION.md). This confirms
geometry cooking and visibility only—not registration, deformation, contact,
or walking.

# Produce one source-derived distal-leg PinJoint URDF for native compiler
# validation. It intentionally has no collision geometry or muscle lowering.
numi human preview --sources Sources --side right --output Build/right-pin-preview

# Preserve and evaluate all Rajagopal CustomJoint function tables as compiler
# IR and default-value test vectors. The matching Core revision executes the
# bounded FunctionBased tree, mobile pelvis root, and source Millard effort in MetalWorld
# free motion or a synthetic streamed-contact probe; this command remains the
# provenance compiler for that source program.
numi human kinematics --sources Sources --output Build/custom-joint-ir

# Compile the entire Rajagopal rigid tree into the provenance-locked Core CPU
# reference payload. This is neither a RobotPack nor an accelerated rollout.
numi human core-reference --sources Sources --output Build/rajagopal-core-reference
```

## What the first manifest means

| Source data | NumiLab Human v1 role | Current physical boundary |
| --- | --- | --- |
| OpenSim bodies, joints, masses, inertias | articulated rigid-body specification | bounded FunctionBased free motion and synthetic source-contact response are device-qualified for the fixed tree and a fail-closed source-default mobile pelvis-root reducer; anatomical registration/contact remain separate |
| OpenSim muscle paths and Hill-type parameters | active muscle–tendon specification | 80 Rajagopal Millard elements accept an explicit control stream or a fail-closed, complete ordered native-task excitation surface, then update activation on device for the fixed tree or source-default mobile root in the synthetic source-contact probe; OpenSim equivalence remains open |
| BodyParts3D bones and muscles | named geometry attached to semantic anatomy | visual/anatomical geometry, not a new independent physical source |
| BodyParts3D skin, organs, vessels, nerves | deformable/anatomical geometry candidates | no material constants or volumetric meshes are supplied upstream |
| tendons, ligaments, cartilage | nonlinear tensile / compliant-contact candidates | only OpenSim tendon parameters are active-source data; all other constitutive data needs a cited calibration |

See [the architecture](Docs/ARCHITECTURE.md), [import procedure](Docs/IMPORT.md),
[bounded execution evidence](Docs/EXECUTION_EVIDENCE.md),
[source-static visual validation](Docs/VISUAL_VALIDATION.md), and
[third-party notices](THIRD_PARTY_NOTICES.md) before building or publishing
derived data.
