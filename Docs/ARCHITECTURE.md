# NumiLab Human v1 architecture

```text
BodyParts3D 4.0                 MyoSim `myofullbody`
named OBJ meshes + FMA trees    103 bodies + 416 active muscle-tendon elements
             \                         /
              \                       /
               native Human source payloads
                         |
     +-------------------+--------------------+
     |                   |                    |
articulated rigid tree active spatial tendon anatomical geometry registry
     |                   |                    |
Core kinematics/dynamics Core J^T scatter     registered visual/deformable candidates
```

## Source-of-truth rules

- **Geometry:** BodyParts3D owns anatomical names, FMA identifiers, OBJ paths,
  and both parent/child hierarchies. Its `BP…` representation identifiers and
  `FJ…` OBJ-element identifiers are distinct, so the importer retains their
  explicit mapping rather than deriving filenames from labels. It never
  supplies mass, inertia, joint centres, activation, or material parameters.
- **Active full-body mechanics:** MyoSim `myofullbody` owns the currently
  active full-body segment topology, source joints, masses/inertias, spatial
  tendon routes, and all 416 MuJoCo muscle parameter records. Core evaluates
  this payload directly in C++; Python is not on the native execution path.
- **Comparative lower-body mechanics:** RajagopalLaiUhlrich2023 owns its own segment
  frames, joints, masses, inertias, muscle geometry paths, and Hill-type
  muscle/tendon values. No geometry-derived inertia may replace it.
- **Upper-body mechanics:** MoBL-ARMS owns its shoulder girdle constraints,
  shoulder/elbow/forearm/wrist DOFs, masses/inertias, and upper-extremity
  muscle paths and parameters. Its right/left naming is retained exactly.
- **No silent registration:** importing creates a registration work item for
  each BodyParts3D mesh-to-OpenSim-body association. A shared word such as
  `femur` is evidence to propose a match, never evidence that coordinates,
  scales, or origins already agree.
- **Cervical/hyoid extension:** Mortensen 2018 supplies a 72-muscle OpenSim 3
  cervical/hyoid source record. It is not force-applied until its rest pose is
  explicitly registered to the active MyoSim neck and an overlapping
  neck/head-body replacement policy is resolved.

## Target mapping

| Numi subsystem | Imported evidence | Required before physical use |
| --- | --- | --- |
| Skeleton | OpenSim rigid segments, joints, mass centres, inertia tensors | bounded fixed-root and source-default mobile-root FunctionBased device execution are qualified; source-frame registration and collision/contact remain open |
| Muscles | OpenSim GeometryPath and Hill-type muscle/tendon fields | bounded source static-equilibrium actuator lowering, including an exact complete native-task excitation surface on the source-default mobile root, is qualified; OpenSim equivalence and held-out force/moment-arm validation remain open |
| Skin | BodyParts3D skin OBJ | shell topology repair and cited constitutive model |
| Organs | BodyParts3D organ surface OBJ | watertight volume mesh plus organ-specific FEM/MPM material model |
| Ligaments | BodyParts3D ligament OBJ, OpenSim coordinate limits where present | attachment paths, nonlinear tensile parameters, and calibration |
| Tendons | OpenSim muscle tendon slack length and path; BodyParts3D tendon geometry when available | path registration and active-muscle force validation |
| Cartilage | BodyParts3D cartilage surface geometry | thickness, compliant-contact law, and validation data |
| Vessels | BodyParts3D vessel geometry | centreline/tube conversion, wall parameters, and fluid/solid scope |
| Nerves | BodyParts3D neural geometry | rendering/annotation semantics only; no neural controller claim |

BodyParts3D itself cautions that some components are artist-made, may overlap,
and represents tube-shaped organs as solids. The importer retains those facts
as explicit conversion gates rather than assigning unsupported physics.

## Numi execution boundary

### Active MyoSim full-body reference

The first source-complete full-body owner path is a Core C++ reference, not a
Python simulator wrapped by Numi. An offline MyoSim composition step writes two
immutable payloads: `NHRIGID2` contains the 157-body Core tree (103 authored
source bodies plus 54 exact zero-inertia transform carriers), while `NHMYO1`
contains all 416 actuator definitions, 1,815 sites, and 143 wrap geometries.
At Core `f564977`, `MujocoMuscleReference` evaluates the MuJoCo general-muscle
activation/force equations and sphere/cylinder spatial tendon routes, scatters
`F * d(length)/d(v)` through Core point Jacobians, and drives the same native
forward-dynamics owner.

`numi human myosim-native-probe Build/myosim-fullbody --metal` executes that
path by directly launching the C++ Core binary, then dispatches the full body's
pose and analytic point-Jacobian stream to Metal. In the same command buffer,
the MyoSim kernel consumes the private pose output to evaluate all 416
sphere/cylinder spatial routes and their default-state static actuator forces;
no CPU-restaged geometry is admitted. On the local Apple M4, the 157-body /
128-DoF source tree passed CPU/GPU parity with maximum body position,
orientation-component, point-position, point-Jacobian, muscle-length, and
muscle-force errors of `6.32e-07 m`, `1.43e-07`, `6.54e-07 m`, `7.35e-07`,
`7.46e-07 m`, and `2.63e-03 N`, respectively, while applying all 90
source-default wraps. No Python process, interpreter-owned physics, or
per-step host loop exists after the payload has been created. The dense
128-DoF mass factor, muscle `J^T` force scatter, and forward-dynamics update
remain Core CPU reference stages today; this does not claim a complete
device-resident muscle-force rollout.

`numi.human.v1` remains an owner-neutral intermediate artifact, but Core
revision `730aba4` now executes the bounded Rajagopal mechanics path: the
source fixed tree and a source-default-preserving physical pelvis mobile-root
reduction run persistent free-motion state on Metal. The mobile reduction
removes only the synthetic anchor, seeds a 7-q/6-v root from the exact source
default pose, and retains the remaining source transforms. The reusable Core
reducer returns canonical source maps and fail-closes on actuator profiles,
geometry/constraints, or moving default roots; it does not equate arbitrary
Euler-coordinate `ground_pelvis` perturbations with free-root state.
All 80 source Millard muscles calculate path tension on device and reduce it
into the same effort arena before each microstep. An optional packed
per-control excitation stream or a fail-closed, complete ordered native-task
action surface advances device activation by an exact first-order hold with
explicit caller-owned time constants. The same bounded path also drives a
synthetic plane/sphere streamed-contact probe. This is a source-mechanics
admission, not a generic external-human RobotPack,
BodyParts3D contact world, or deformable-body claim.

The tracked workspace command `numi human` is the bridge at this stage. It
uses the normal Numi capability discovery path to fetch and compile this
repository's source-faithful artifact; it cannot register a robot or schedule
a rollout until the core lowerer exists and all gated sources are supplied.

`numi human audit` records the inspected Numi runtime contract alongside the
imported lower-body source. At runtime revision `730aba4`, the Core preserves
the canonical source program, evaluates its source-order pose/motion subspace,
and advances the bounded FunctionBased state through MetalWorld's resident
`q`/`v`/effort arenas, including the synthetic source-contact probe and the
source-default mobile-root reduction.

`numi human core-reference` now compiles the complete 22-body Rajagopal tree
into a fixed-layout payload: each source body supplies its mass, COM, and
inertia; each source joint supplies its resolved body-frame anchors and
rotations; every scalar coordinate is retained; and all 10 CustomJoint
programs are decoded into the same Core model. The loader then executes
kinematics, a positive-definite FP64 mass matrix, inverse dynamics, forward
dynamics, and invariant evaluation. This establishes an executable
source-faithful *rigid-tree CPU reference*, not BodyParts3D registration,
collision/contact, muscle actuation, an OpenSim numerical equivalence study,
or whole-human physical validation.

The bounded Metal articulated operator consumes the program for the whole
Rajagopal tree: source poses, point Jacobians, and dense mass assembly run on
device. MetalWorld then runs the same FunctionBased kinematics/Jacobians,
source-materialized Millard static fiber-tendon equilibrium, optional packed
or exact complete native-task first-order activation, finite-cylinder
GeometryPaths, and a deterministic per-DoF force reduction in one command
buffer before the persistent source-dynamics step or its synthetic streamed
contact response. The mobile-root reducer remaps body-local source path/wrap
identity after its synthetic-anchor removal and proves default-pose continuity;
it is not an OpenSim binary-equivalence result,
hybrid wrap-history implementation, anatomical contact claim, or material
calibration.

Rajagopal's `radius_hand_r` and `radius_hand_l` are a narrower case: both
source UniversalJoint coordinates are explicitly locked at a zero default, so
their source transform is exactly the fixed zero-coordinate transform. The
importer records those two fixed lowerings individually; it does not claim
that arbitrary movable UniversalJoints are supported.

The importer retains every `TransformAxis` function and XML subtree, all
muscle curve parameters, every path-point and wrap subtree, and each wrap
object subtree. A source-faithful Metal extension therefore has the source
data it needs; a provisional preview must identify each deliberately reduced
joint, collision, or actuator contract instead of overwriting this record.

`numi human kinematics` additionally emits the exact Rajagopal CustomJoint
function tables and default/unit-velocity pose, `H`, and `Hdot` test vectors,
plus canonical `MROpenSimSpatialTransformGPU` binary programs and fixed-state
sidecars. This is compiler IR for a function-based articulation extension; the
matching Metal evaluator is a kinematic boundary, not an articulated solver.

`numi human muscles` emits a separately validated Millard source IR: the
parameters, curve subtrees, GeometryPath points, PathWrap records, and wrap
objects remain source-faithful and frame-resolved. `numi human
millard-reference` compiles its static-reference companion payload, including
the documented OpenSim defaults for omitted optional curve properties. The
owner Core executes its source force projection inside MetalWorld; pinned
OpenSim parity and held-out force/moment-arm validation remain separate gates.
