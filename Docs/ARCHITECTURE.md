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
At Core `86790f3`, `MujocoMuscleReference` evaluates the MuJoCo general-muscle
activation/force equations and sphere/cylinder spatial tendon routes, scatters
`F * d(length)/d(v)` through Core point Jacobians, and drives the same native
forward-dynamics owner.

The full-body reference also makes that force-to-state coupling executable: an
otherwise identical passive and 416-muscle free-body state are each integrated
for 1 µs in the FP64 Core. The muscle-driven state differs by maximum velocity
`0.0714839058782` and configuration `7.14839058782e-08`. This is deliberately
a short unconstrained sensitivity check, not an assertion that the source
default co-activation is a stable posture, realistic gait, or contact result.

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

The original `numi human myosim-native-visuals` capture drew every route as a
straight site/wrap-centre segment. That output is retained only as a coverage
regression because it can cross a wrap and does not depict a tendon. The
current native inspection path still receives a Metal-produced 157-body pose,
but leaves route geometry hidden by default. Its opt-in focused diagnostic
evaluates the same posed source route on the CPU reference, then emits exact
site endpoints, tangent contacts, and sampled sphere/cylinder wrap arcs. The
host publication and CPU route-resolution boundaries mean this is neither a
device-resident live presentation path nor tendon-surface or BodyParts3D
attachment geometry.

Core `2aab522` consumes the explicitly provisional `NHBONES1` visual input.
The offline importer records exact BodyParts3D triangles and one decomposed
uniform-scale rest transform for each selected major bone. Its 18 unambiguous
segment meshes establish the source-to-MyoSim common-frame fit; nine additional
named meshes—bilateral hip bones, fibulae, tali, patellae, and sternum body—use
that fixed common frame and their conservative parent links. The native visual
probe validates the payload against the same MyoSim archive identity, attaches
all 27 instances to Core articulated-link indices, then uses the Metal pose
snapshot for rendering. The M4 Pro evidence covers 56,995 vertices and 322,074
indices plus the complete route/site overlay. This is an executable body-frame
visual binding, but it remains outside physical admission: its common-frame
centroid/COM fit cannot stand in for anatomical landmarks, collision
calibration, skinning, or tissue models. In particular, the active MyoSim
source has no separate fibular rigid body, so each fibula remains a visual
attachment to the ipsilateral tibial link rather than a claimed independent
articulated segment.

The focused posterior-calf surface package is a separate `NHTISS2` input. It
contains the exact BodyParts3D gastrocnemius, soleus, and calcaneal-tendon
triangles, two named MyoSim body-frame rest transforms per surface, and one
0–1 proximal weight per source vertex. The native renderer evaluates each
vertex through both posed bodies and linearly blends the resulting positions
and normals into a world-surface snapshot. Thus gastrocnemius spans
femur-to-calcaneus, while soleus and calcaneal tendon span tibia-to-calcaneus;
it fixes the invalid one-rigid-parent representation of a crossing tendon.
For the calcaneal-tendon insertion, source vertices within 3 mm of the named
BodyParts3D calcaneus mesh are locked to the calcaneus parent and the next
12 mm is feathered before this blend. That prevents the driven visual tendon
from being pulled off the calcaneus by a body-centre-derived weight; it is
still explicit presentation skinning only—not FEM/MPM, muscle-fibre
contraction, tendon constitutive response, collision, force transfer, or a
biological attachment certificate.

The full-body `NHTISS2` package uses the same native ABI but is generated from
`bodyparts3d-myosim-surface-map.v1.json`. Each normal row is source-name
validated, then resolves its two parents from the first and final sites of its
named compiled MyoSim actuator route. A row that names a partitioned muscle
must have agreeing route endpoint pairs; it fails closed otherwise. The two
calcaneal tendon rows are the explicit shared-tendon exception and identify
their contributing gastrocnemius/soleus routes while binding the surface from
tibia to calcaneus. This is stronger source ownership than visual proximity,
but still only pose-driven linear-blend rendering; `NHMYO1` remains the force
path authority.

Core `2aab522` adds a separate, opt-in bounded muscle-driven visual state.
`myosim-native-muscle-bone-visuals` evaluates the complete 416 MyoSim route
definitions at source-default excitation/activation (`0.5` / `0.5`) in the
existing FP64 Core reference, scatters the resulting generalized muscle force,
and compares a one-step free-body integration to an identically integrated
passive state. Only the resulting active configuration is supplied to the
Metal articulated operator; Metal owns final pose computation and rendering,
not the force projection or dense forward-dynamics integration. The command
allows only a 1 µs–1 ms step and defaults to 1 ms for an inspectable
sensitivity capture. That boundary is intentional: the 1 ms all-muscle
co-activation probe differs from passive by `71.4839058782` maximum velocity
and `0.0714839058782` maximum configuration, so it demonstrates native
force-to-pose coupling but cannot be interpreted as a stable stance, control
policy, contact result, gait, or physiological prediction.

The supported capture adds a narrow, source-owned contact layer. `NHCNT1`
serializes MyoSim's compiled plane and its five authored capsule/ellipsoid
foot witnesses per side. The default pose begins about 53.2 mm above that
plane, so the capture derives a ground-aligned seed from the minimum witness
gap and retains the two bilateral lowest witnesses. After the all-muscle free
velocity step, Core's FP64 exact-cone solver resolves that support and Core
advances the configuration; Metal then owns the articulated pose and renderer.
The current 157-body connected tree is not admitted to the fixed Metal
full-dynamics contact bucket, and the executable reports that status rather
than substituting a GPU claim. This bounded two-witness snapshot is not
general collision, stable posture, gait, compliance calibration, or a
deformable-tissue solve.

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

`numi human audit` records the active MyoSim full-body route separately from
the legacy stitched source manifest and verifies the inspected runtime checkout.
At Core `86790f3`, it reports the Apple M4 parity evidence for the 157-body /
416-muscle route-force reference while retaining the bounded Rajagopal
FunctionBased execution as comparative lower-body mechanics. It does not
present the unavailable authenticated MoBL-ARMS archive as a blocker to the
active full-body MyoSim route.

When selected explicitly, the audit also exposes
`free_human_foundation_source_stack`: verified BodyParts3D 4.0 anatomy,
RajagopalLaiUhlrich2023 lower-body mechanics, and the pinned public CEINMS
MoBL-ARMS 4.1 upper-extremity model. It is ready for a source import with a
non-commercial **unimanual** upper-body variant. That is intentionally a
different claim from the original authenticated bimanual archive, and it does
not imply bilateral upper-body completion, body-frame registration, or a
physical qualification.

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
