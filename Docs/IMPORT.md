# Import procedure

## 0. Active full-body source to native Core payload

The primary full-body route is the pinned Apache-2.0 MyoSim composition, not
the older lower-/upper-body stitching sequence below. Its upstream composition
adapter is deliberately offline; after it has produced the payloads, Numi
execution is a shell-dispatched native C++ Core process with no Python runtime.

```sh
# Offline source acquisition and compilation.
numi human myosim-fetch --output Sources
numi human myosim-build \
  --sources Sources \
  --python /path/to/source-only-myosim-python \
  --output Build/myosim-fullbody

# Native Numi Core execution plus Apple-GPU full-body pose/Jacobian,
# muscle-route, and static-force parity.
numi human myosim-native-probe Build/myosim-fullbody --metal

# Optional Apple-native default-pose proxy capture. Routes are hidden by
# default because an all-muscle overlay is not anatomy presentation.
# This is a native executable; no Python process is started.
numi human myosim-native-visuals \
  Build/myosim-fullbody \
  Build/myosim-fullbody/native-articulated-visuals

# Offline BodyParts3D source import: derive a reviewable visual-skeleton
# rest-frame candidate, then write compact exact-triangle input for the native
# renderer. The attachment refinement is optional and intended for focused
# source-site diagnostics, not the broad visual-skeleton package.
numi human myosim-bodyparts-registration \
  --sources Sources \
  --artifact Build/myosim-fullbody \
  --output Build/myosim-fullbody/bodyparts3d-major-bone-registration.candidate.json
numi human myosim-bodyparts-bone-payload \
  --sources Sources \
  --registration Build/myosim-fullbody/bodyparts3d-major-bone-registration.candidate.json \
  --output Build/bodyparts3d-myosim-major-bones

# Native C++/Metal capture: no Python process is started here.
numi human myosim-native-bone-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-major-bones/native-articulated-full-skeleton-180-views \
  --dimension 2048

# Focused native path diagnostic: MyoSim indices 348/349/369/371 are right
# gastrocnemius lateralis/medialis, soleus, and tibialis anterior; body 136 is
# its right tibial link. The renderer resolves tangent contacts and wrap arcs,
# then projects only source-site endpoints to matching BodyParts3D bone
# triangles for the diagnostic image; force/path evaluation is unchanged.
numi human myosim-native-route-inspection \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/route-inspection-right-lower-leg \
  136 348 349 369 371

# Native bounded force-to-pose capture: Core FP64 projects all 416 source
# muscles, integrates one free-body sensitivity step, and Metal renders only
# its final articulated pose. No Python process is started.
numi human myosim-native-muscle-bone-visuals \
  Build/myosim-fullbody \
  Build/bodyparts3d-myosim-major-bones/bodyparts3d-myosim-major-bones.nhbones \
  Build/bodyparts3d-myosim-major-bones/native-muscle-driven-full-skeleton-180-views \
  --muscle-step-seconds 0.001 --dimension 2048
```

The native probe validates 416 full-body muscle-tendon elements and their
source route geometry at the source default pose. `--metal` additionally
validates the entire 157-body / 128-DoF pose and analytic point-Jacobian stream
plus all 416 MuJoCo spatial routes and default-state static actuator forces on
Apple GPU against Core. Muscle `J^T` scatter and dense forward dynamics remain
CPU-reference execution, not a Metal rollout. The selected Mortensen
2018 neck source is imported separately with `numi human mortensen-neck`; do
not attach its 72 muscles to the active MyoSim body before an explicit
rest-pose registration. See
[visual progress](VISUAL_PROGRESS.md) for the inspected full-body views and
[architecture](ARCHITECTURE.md) for runtime ownership.

The native marker visual command emits four fixed cameras plus a Core visual-pack and
manifest. It binds inertial-body proxies to the default Metal articulated-pose
snapshot and hides all route lines. Use `myosim-native-route-inspection` with
an explicit small muscle set when inspecting tangent/arc path geometry. Its
endpoint-to-bone projection is visual-only, and it is proof of this snapshot
render chain only; BodyParts3D registration and live device-resident
presentation remain separate work.

The bone-payload importer is also offline source preparation. It checks the
selected archive/member identities, converts only exact OBJ triangles and
triangle-derived normals, and carries one uniform-scale local transform per
bone. `myosim-native-bone-visuals` consumes that compact payload directly in
the native executable, binding 180 named source meshes to the Metal pose with
routes hidden by default. Seventeen conservative major-bone landmarks own the
common-frame fit; the cranial bones, ribs, hands, feet, and axial meshes inherit
that frame and bind to their named MyoSim parent. Where MyoSim provides only a
whole torso or toes parent, the corresponding ribs or mid-foot/toe source meshes
share that parent rather than being granted invented individual mechanics. The
candidate is visual-only: it does not create colliders, contact constants,
skinning weights, soft tissue, or a medical registration result.

`myosim-native-muscle-bone-visuals` is a visual sensitivity command, not a
rollout interface. It only permits a 1 µs–1 ms step (default 1 ms), projects
all 416 `NHMYO1` source muscles at their deterministic source-default
activation/excitation, and compares the active configuration to a matched
passive step before the final Metal pose/render pass. The command does not
create a controller, keep force/dynamics resident on Metal, add contact, or
validate standing or gait.

## 1. Fetch the password-free sources

```sh
numi human fetch --output Sources
```

This fetches the six official BodyParts3D hierarchy/definition tables, the two
official 4.0 OBJ archives, and the RajagopalLaiUhlrich2023 model pinned in
`sources.lock.json`. Files with a known SHA-256 are verified before use.

## 2. Download MoBL-ARMS yourself

Sign in at the official [Upper Extremity Dynamic Model
page](https://simtk.org/projects/upexdyn/) and download its official bimanual
release, `MobL_ARMS_OpenSim3_bimanual_model.zip`. Keep the original archive
unchanged. Do not substitute a third-party GitHub mirror: this importer makes
the source URL and licence gate part of the output fingerprint.

## 3. Build an audit artifact

```sh
numi human build \
  --sources Sources \
  --upper-archive /absolute/path/MobL_ARMS_OpenSim3_bimanual_model.zip \
  --accept-upper-noncommercial-terms \
  --output Build/human-v1
```

The command emits:

- `human.v1.json` — source-preserving Numi Human v1 intermediate manifest.
- `report.json` — counts, source hashes, available parameter fields, and
  unresolved geometry registrations.

Both output directories are ignored because they are derived from source data.
Re-run the command from clean third-party archives when a new import revision
is required.

## 4. Audit every gate

```sh
numi human audit \
  --sources Sources \
  --runtime-root /absolute/path/to/MetalRobo \
  --output Build/human-v1-gates.json
```

The gate report distinguishes the active pinned Apache-2.0 MyoSim full-body
route from the legacy BodyParts3D + Rajagopal + authenticated MoBL-ARMS
manifest. It reports source readiness, the current Core contract, and the
separate evidence boundaries for device route-force parity, `J^T` scatter,
forward dynamics, anatomical registration, and material validation. It never
promotes an open gate based on a naming match or a successful JSON build.

When `--runtime-root` is supplied, it also records whether that checkout is
clean and at the exact runtime revision whose lowering capabilities were
audited. A missing, dirty, or revision-mismatched checkout is not runtime
evidence.

## 5. Preflight every BodyParts3D OBJ member

```sh
numi human geometry-audit \
  --sources Sources \
  --output Build/bodyparts3d-topology.json
```

This writes the exact archive/member name and SHA-256 for every OBJ, with raw
vertex/face counts, bounds, and conservative edge-manifold facts. It does not
repair a mesh, establish an anatomical frame registration, create a volume
mesh, or infer a material law. Those remain separate, source-specific gates.

## 6. Export a source-static BodyParts3D visual preview

```sh
numi human visual-preview \
  --sources Sources \
  --output Build/bodyparts3d-skin-preview
```

The default member is the exact `FJ2810` full-skin OBJ. The command emits a
self-contained GLB with millimetres converted to metres and a provenance
manifest containing the archive, member, and generated-file hashes. It is an
inspection artifact only: no BodyParts3D-to-OpenSim frame registration, skin
deformation, collision, material property, or Human RobotPack is inferred.

At the matching Core revision, cook the GLB and render its three static
inspection cameras with `metalrobo_visual_cook` and
`metalrobo_bodyparts3d_visual_probe`. The probe emits PPM frames; retain their
hashes with the source member and device log. Do not publish those frames as a
physically actuated Human result.

### Right lower-leg source anatomy reference

The focused native MyoSim route diagnostic is deliberately a centreline tool,
not a tendon mesh. Export the exact source-static BodyParts3D alternative when
reviewing visible lower-leg anatomy:

```sh
numi human right-lower-leg-anatomy-preview \
  --sources Sources \
  --output Build/bodyparts3d-right-lower-leg-anatomy

$NUMI_LAB_ROOT/build/bin/metalrobo_visual_cook \
  Build/bodyparts3d-right-lower-leg-anatomy/bodyparts3d-right-lower-leg-anatomy-source-static.glb \
  Build/bodyparts3d-right-lower-leg-anatomy/bodyparts3d-right-lower-leg-anatomy-source-static.mrvpack \
  --id bodyparts3d-right-lower-leg-anatomy-source-static \
  --license CC-BY-4.0 \
  --provenance 'BodyParts3D 4.0 exact source-static right lower-leg bundle'

$NUMI_LAB_ROOT/build/bin/metalrobo_bodyparts3d_visual_probe \
  Build/bodyparts3d-right-lower-leg-anatomy/bodyparts3d-right-lower-leg-anatomy-source-static.mrvpack \
  Build/bodyparts3d-right-lower-leg-anatomy/native-views \
  --dimension 1024 --focus-lower-third
```

The bundle contains exact right femur/patella/tibia/fibula/talus/calcaneus,
gastrocnemius heads, soleus, tibialis anterior, and calcaneal-tendon source
surfaces. Its three semantic preview materials distinguish source bone, muscle,
and tendon layers; they do not add tissue parameters. All components share the
BodyParts3D rest frame, but that fact is not a MyoSim attachment transform,
skinning map, collision admission, or dynamic tendon claim.

### Focused calcaneal-tendon source inspection

For a readable posterior-chain inspection, use the smaller exact source bundle
below. It retains the source OBJ vertex normals where their face mapping is
compatible and omits unrelated anterior geometry that otherwise hides the
calcaneal tendon. Its source-mesh proximity report is descriptive only.

```sh
numi human right-calcaneal-tendon-continuity-preview \
  --sources Sources \
  --output Build/bodyparts3d-right-calcaneal-tendon-continuity

$NUMI_LAB_ROOT/build/bin/metalrobo_visual_cook \
  Build/bodyparts3d-right-calcaneal-tendon-continuity/bodyparts3d-right-calcaneal-tendon-continuity-source-static.glb \
  Build/bodyparts3d-right-calcaneal-tendon-continuity/bodyparts3d-right-calcaneal-tendon-continuity-source-static.mrvpack \
  --id bodyparts3d-right-calcaneal-tendon-continuity-source-static \
  --license CC-BY-4.0 \
  --provenance 'BodyParts3D 4.0 authored-normal posterior-chain source inspection'

$NUMI_LAB_ROOT/build/bin/metalrobo_bodyparts3d_visual_probe \
  Build/bodyparts3d-right-calcaneal-tendon-continuity/bodyparts3d-right-calcaneal-tendon-continuity-source-static.mrvpack \
  Build/bodyparts3d-right-calcaneal-tendon-continuity/native-views \
  --dimension 2048 --fill-frame
```

## 7. Compile a limited source-derived distal-leg preview

```sh
numi human preview \
  --sources Sources \
  --side right \
  --output Build/right-pin-preview
```

This output contains the right tibia, talus, calcaneus, and toes with their
Rajagopal mass/inertia data and the exact supported ankle, subtalar, and MTP
PinJoint transforms. It purposefully contains no collision geometry and no
muscle lowering, so it is only a native imported-URDF compiler preview—not a
complete Human RobotPack or physical validation.

When the matching Numi runtime is available, the bounded Metal ABA check is:

```sh
metalrobo_robot_description_cooker_probe \
  --metal Build/right-pin-preview/rajagopal-right-distal-pin-preview.urdf
```

It reports the Metal device, a successful GPU status, and a numerical payload
fingerprint. Repeat the same invocation on the same binary and device before
calling it a deterministic replay. This proves neither collision, contact,
muscle actuation, nor full-human physics.

## 8. Preserve the source CustomJoint functions for the core lowerer

```sh
numi human kinematics \
  --sources Sources \
  --output Build/custom-joint-ir
```

The resulting IR includes all 10 Rajagopal `CustomJoint` SpatialTransforms,
their `Constant`, `LinearFunction`, `PolynomialFunction`, and `SimmSpline`
tables, plus source-order pose, motion-subspace `H`, and `Hdot` default and
unit-velocity test vectors. It also writes one 2,512-byte canonical
`MROpenSimSpatialTransformGPU` program per joint and 64-byte input sidecars
under `opensim-spatial-programs/`; the manifest hashes every file.

At the pinned Core revision, check one source-derived program on an Apple
Metal device with:

```sh
metalrobo_opensim_spatial_transform_gpu_probe \
  --program Build/custom-joint-ir/opensim-spatial-programs/walker_knee_r.mrospatial \
  --input Build/custom-joint-ir/opensim-spatial-programs/walker_knee_r.default.mrospatialinput
```

The probe rejects a non-canonical binary, evaluates the decoded program on
CPU and GPU, compares pose/`H`/`Hdot` within FP32 tolerance, and repeats the
GPU result byte-for-byte. The pinned FP64 reference can consume an immutable
FunctionBased program in its analytic mass/bias and state path, but this
command does not assemble the Rajagopal skeleton into that model. Metal ABA
does not yet consume this IR; none of this substitutes for the human lowerer
or physical validation.

## 9. Export BodyParts3D nerve annotations

```sh
numi human nerve-annotations \
  --sources Sources \
  --output Build/bodyparts3d-nerve-annotations.json
```

This retains the selected nerve labels, `FJ...` mesh references, and incident
`is_a`/`part_of` source hierarchy edges. It is deliberately annotation-only;
no neural conduction, activation, collision, or deformable model is inferred
from BodyParts3D geometry.

## 10. Preserve the Rajagopal Millard muscle source contract

```sh
numi human muscles \
  --sources Sources \
  --output Build/rajagopal-millard-muscle-ir.json
```

The IR validates and retains all 80 `Millard2012EquilibriumMuscle` records,
their source parameters and curve subtrees, 288 body-frame path points, 46
path-wrap references, and 44 source wrap objects. It does not evaluate a
Hill-type force or apply one to a Numi coordinate by itself; use the matching
reference artifact below for the bounded static-equilibrium path.

## 10a. Compile the Rajagopal Millard reference payload

```sh
numi human millard-reference \
  --sources Sources \
  --output Build/rajagopal-millard-reference
```

This provenance-locked `NHMUSC1` ABI v3 payload binds to the matching rigid
tree source hash and carries all 80 source muscle scalars, 22 source curve
parameters per muscle, 288 COM-relative path points, and 46 finite-cylinder
wrap definitions with their source `PathWrap` method and 1-based range.
Optional source curve properties that are absent in the model are materialized
only with the documented OpenSim class defaults.

At the matching Core revision, run both the source tree and its muscle
reference together:

```sh
metalrobo_numilab_human_core_reference_probe \
  Build/rajagopal-core-reference/rajagopal-core-reference.nhrigid \
  --metal \
  --millard Build/rajagopal-millard-reference/rajagopal-millard-reference.nhmuscle
```

The bounded FP64 reference reconstructs the source quintic-Bezier curves,
solves static fiber-tendon equilibrium at the imported reference pose, evaluates
the finite-cylinder GeometryPaths, and scatters tensile path force through the
articulated point Jacobians. With both `--metal` and `--millard`, the same
invocation also runs the bounded MetalWorld path: source poses/Jacobians,
Millard force projection, deterministic generalized-force reduction, and the
fixed-root FunctionBased state step plus the Core-owned
`reduceFixedFunctionBasedRootToMobileDefaultPose` source-default mobile-root
reduction remain in one command buffer.
This is not OpenSim binary-equivalence, a full implementation of OpenSim's
hybrid wrap-history behavior, contact, or material validation; the source
PathWrap XML, including method and range, remains in the companion IR for that
later equivalence gate.

## 11. Resolve the Rajagopal rigid-body tree

```sh
numi human skeleton \
  --sources Sources \
  --output Build/rajagopal-rigid-skeleton-ir.json
```

This resolves every OpenSim joint socket through its local frame chain to the
22 source rigid bodies, retains all mass/inertia, frames, coordinates, and
motion axes, and links each CustomJoint to its canonical program filename. It
records 10 scalar-core-supported PinJoints, two exact locked-fixed wrists, and
10 FunctionBased joints. It is not a body-to-BodyParts3D registration or
collider cook.

## 12. Compile and run the complete Rajagopal CPU reference tree

```sh
numi human core-reference \
  --sources Sources \
  --output Build/rajagopal-core-reference
```

This writes `rajagopal-core-reference.nhrigid` plus a SHA-256 manifest. The
binary has a documented little-endian ABI containing a synthetic fixed root,
the 22 source bodies, 22 joints, 35 coordinates/velocities, and all 10
canonical FunctionBased programs. It is intentionally source-locked rather
than a general interchange format.

At the exact matching Core revision, execute its bounded FP64 reference gate:

```sh
metalrobo_numilab_human_core_reference_probe \
  Build/rajagopal-core-reference/rajagopal-core-reference.nhrigid --metal
```

The probe rejects an incompatible or trailing-byte payload, validates Core
ownership rules, evaluates whole-tree kinematics and the analytic mass matrix,
then requires forward dynamics to recover a source-state inverse-dynamics
acceleration. With `--metal`, it additionally runs the bounded persistent
FunctionBased MetalWorld state step against the FP64 reference. Supplying a
matching `--millard` payload adds 80 source muscle forces to MetalWorld's
resident effort arena and verifies their aggregate force and resulting
acceleration against the FP64 bridge. The probe also runs a synthetic
plane/sphere contact witness; it does not compare against an OpenSim runtime,
register BodyParts3D meshes, admit anatomical collision/contact, or qualify
deformable anatomy.
