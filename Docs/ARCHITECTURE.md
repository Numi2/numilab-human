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
  and both parent/child hierarchies. It never supplies mass, inertia, joint
  centres, activation, or material parameters.
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
| Skeleton | OpenSim rigid segments, joints, mass centres, inertia tensors | source-frame to Numi-frame registration, collision proxies, and compiled-run validation |
| Muscles | OpenSim GeometryPath and Hill-type muscle/tendon fields | Metal muscle-tendon actuator lowering and force/length validation |
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

The live Numi runtime currently supplies built-in robot packs, but not a generic
external-human pack loader. `numi.human.v1` is deliberately an owner-neutral
intermediate artifact. The follow-on Numi core change must lower its selected
mechanics into a `RobotPack`, preserve source hashes in the compiled-run
fingerprint, execute all active muscle-tendon elements in Metal, and validate
the resulting contact/force behavior. Until then, generated manifests are
integration inputs, not runnable human dynamics.

The tracked workspace command `numi human` is the bridge at this stage. It
uses the normal Numi capability discovery path to fetch and compile this
repository's source-faithful artifact; it cannot register a robot or schedule
a rollout until the core lowerer exists and all gated sources are supplied.
