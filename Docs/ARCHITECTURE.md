# NumiLab Human v1 architecture

```text
BodyParts3D 4.0                 OpenSim RLU 2023 + MoBL-ARMS
named OBJ meshes + FMA trees    bodies + joints + inertias + muscle paths
             \                         /
              \                       /
               numi.human.v1 source manifest
                         |
     +-------------------+--------------------+
     |                   |                    |
articulated skeleton   active muscle-tendon  anatomical geometry registry
     |                   |                    |
Numi RobotPack          Numi actuator        Numi visual/deformable candidates
```

## Source-of-truth rules

- **Geometry:** BodyParts3D owns anatomical names, FMA identifiers, OBJ paths,
  and both parent/child hierarchies. Its `BP…` representation identifiers and
  `FJ…` OBJ-element identifiers are distinct, so the importer retains their
  explicit mapping rather than deriving filenames from labels. It never
  supplies mass, inertia, joint centres, activation, or material parameters.
- **Lower-body mechanics:** RajagopalLaiUhlrich2023 owns its own segment
  frames, joints, masses, inertias, muscle geometry paths, and Hill-type
  muscle/tendon values. No geometry-derived inertia may replace it.
- **Upper-body mechanics:** MoBL-ARMS owns its shoulder girdle constraints,
  shoulder/elbow/forearm/wrist DOFs, masses/inertias, and upper-extremity
  muscle paths and parameters. Its right/left naming is retained exactly.
- **No silent registration:** importing creates a registration work item for
  each BodyParts3D mesh-to-OpenSim-body association. A shared word such as
  `femur` is evidence to propose a match, never evidence that coordinates,
  scales, or origins already agree.

## Target mapping

| Numi subsystem | Imported evidence | Required before physical use |
| --- | --- | --- |
| Skeleton | OpenSim rigid segments, joints, mass centres, inertia tensors | bounded fixed-root FunctionBased device execution is qualified; source-frame registration and collision/contact remain open |
| Muscles | OpenSim GeometryPath and Hill-type muscle/tendon fields | bounded source static-equilibrium actuator lowering is qualified; OpenSim equivalence and held-out force/moment-arm validation remain open |
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

`numi.human.v1` remains an owner-neutral intermediate artifact, but Core
revision `7d3b87c` now executes the bounded Rajagopal mechanics path: one
fixed-root FunctionBased articulation runs persistent free-motion state on
Metal, and 80 source Millard muscles calculate path tension on device and
reduce it into that same effort arena before each microstep. The same bounded
path also drives a synthetic plane/sphere streamed-contact probe. This is a
source-mechanics admission, not a generic external-human RobotPack,
BodyParts3D contact world, or deformable-body claim.

The tracked workspace command `numi human` is the bridge at this stage. It
uses the normal Numi capability discovery path to fetch and compile this
repository's source-faithful artifact; it cannot register a robot or schedule
a rollout until the core lowerer exists and all gated sources are supplied.

`numi human audit` records the inspected Numi runtime contract alongside the
imported lower-body source. At runtime revision `7d3b87c`, the Core preserves
the canonical source program, evaluates its source-order pose/motion subspace,
and advances the bounded fixed-root FunctionBased state through MetalWorld's
resident `q`/`v`/effort arenas, including the synthetic source-contact probe.

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
source-materialized Millard static fiber-tendon equilibrium, finite-cylinder
GeometryPaths, and a deterministic per-DoF force reduction in one command
buffer before the persistent source-dynamics step or its synthetic streamed
contact response. This is not an OpenSim binary-equivalence result,
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
