# Bounded execution evidence

## Complete locally available source-artifact sweep — 2026-08-26

All locally available, provenance-locked source compilers were rerun from the
two BodyParts3D 4.0 archives and the pinned Rajagopal model. This is inventory
and import evidence; it does not turn source surfaces into calibrated physical
objects.

| Artifact | Result |
| --- | --- |
| BodyParts3D OBJ preflight | 3,492 meshes; 6,055,747 valid vertices; 9,820,026 triangular faces; 227 closed 2-manifold candidates and 3,265 open surfaces; no invalid OBJ face or vertex references |
| Neural annotations | 101 named nerve-surface components and 101 source hierarchy edges, annotation-only |
| Rajagopal FunctionBased IR | 10 CustomJoints and 10 canonical programs: 21 Constant, 23 LinearFunction, 10 PolynomialFunction, and 6 SimmSpline axes |
| Rajagopal Millard IR | 80 muscles, 288 source body-frame path points, 46 PathWrap records, and 44 WrapCylinder definitions |
| Distal-leg preview | 4 source bodies and 3 source PinJoints; intentionally excludes all 80 muscles and all BodyParts3D collision geometry |

The source sweep does not include MoBL-ARMS because the original authenticated
SimTK archive has not been supplied. It also does not repair the 3,265
open-surface candidates, infer BodyParts3D-to-OpenSim frames, or assign tissue
material constants. Those are explicit evidence gates rather than recoverable
metadata omissions.

## Full Rajagopal rigid-tree CPU reference and device-program regression — 2026-08-26

This establishes one executable source-faithful rigid-body *reference* tree.
It is not a Metal FunctionBased ABA run, a BodyParts3D registration, contact,
muscle actuation, deformable anatomy, calibration, or an OpenSim numerical
equivalence study.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Core revision | `47a3a2e80a4d49506377304a4cfc7315388eff45` |
| Isolated Mini checkout | `/Users/n/MetalRobo-numilab-human-core-reference-47a3a2e` |
| Device / toolchain | Apple M4 Pro on `macmini`; AppleClang `21.0.0.21000101` |
| Reference payload | 22 source bodies, 23 engine bodies (one synthetic fixed root), 22 joints, 35 `q`, 35 `v`, 10 FunctionBased programs |
| Payload SHA-256 | `da7e52ddd64728ed0a63e73a11cf857ec5489b3eb29e32d11f352f35507cdee6` |
| Core-reference probe SHA-256 | `7442907040afbce05b6cd7efae68d728c18f4fb34f53a2145e0ccd8524ca84b3` |
| GPU-program probe SHA-256 | `0e34cff18f33bb300d77ede410310a2445b1e0ed06bd817e9688d01896efd4ec` |
| Metal library SHA-256 | `673c798ae28f1cb92803c58114fdfe1f003ac0b4e2bc29d0c0390ab2966d6619` |
| GPU sweep log SHA-256 | `0846615cd4d416cff5c94ffa531bac6e6abd8642963365c931b90048bb860385` |

`metalrobo_numilab_human_core_reference_probe` loaded the exact payload on the
Mini and completed whole-tree kinematics, a finite symmetric positive-definite
mass matrix (minimum Cholesky pivot `1.578273e-02`), inverse dynamics, forward
dynamics, and invariants. The prescribed inverse-to-forward recovery error was
`4.438117e-14`. That demonstrates the payload is admitted to the actual FP64
Core reference solver, but it neither proves agreement with an OpenSim
trajectory nor validates anatomical geometry registration or tissue physics.

The same isolated checkout rebuilt the device probe and ran every 35 existing
Rajagopal FunctionBased program/input sidecar on the Apple GPU. All 35 passed
the canonical decode/re-pack, source-order pose/`H`/`Hdot`, wrench projection,
and `Hdot*qdot` checks. The current Metal ABA/operator/MetalWorld paths still
explicitly reject FunctionBased joints; this sweep is device-program evidence,
not a full-tree accelerated solve.

## FunctionBased CPU reference and device-program regression — 2026-08-26

This is a bounded Core contract for one source-derived FunctionBased joint and
the canonical device-program suite. It is not an assembled Rajagopal human,
contact qualification, muscle actuation, or experimental validation.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Numi runtime revision | `f1c7ac5aa609e37196720d13f0f9011f59d29cc1` |
| Isolated Mini checkout | `/Users/n/MetalRobo-numilab-function-f1c7ac5` |
| Device | Apple M4 Pro on `macmini` |
| Toolchain | AppleClang `21.0.0.21000101` |
| Source program package | 10 canonical 2,512-byte programs and 35 canonical 64-byte state inputs |
| GPU-probe SHA-256 | `a8f2f11d00394e30222403574356ffe0e68c729983f19effb0cb41ef2fb11cde` |
| Metal-library SHA-256 | `673c798ae28f1cb92803c58114fdfe1f003ac0b4e2bc29d0c0390ab2966d6619` |

`metalrobo_articulated_dynamics_probe` passed on the Mini. Its source-derived
`walker_knee_r` FunctionBased check reproduced the source joint-frame
translation, `H*qdot`, and linear Jacobian with error `2.710505e-18`, then
closed inverse-to-forward acceleration with error `2.220446e-16`. The model is
a two-body FP64 reference fixture using the real joint function; it is not the
22-body human or a calibrated inertial validation.

The same isolated checkout ran all 35 Rajagopal program/input sidecars through
the existing Metal pose/`H`/`Hdot`, source-wrench projection, and `Hdot*qdot`
device path. Every case passed; a representative `hip_r` run reported
`device=Apple M4 Pro` and `tau0=0.41`. Current Metal ABA/operator/MetalWorld
kernels explicitly reject FunctionBased joints, so this does not make the
source skeleton GPU-executable.

## FunctionBased source-wrench projection — 2026-08-26

This extends the prior source-sidecar kinematic proof with a bounded
generalized-force primitive. It is not multi-body articulated dynamics,
contact, muscle actuation, or full-human evidence.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Numi runtime revision | `58dc262977092e63bd0f73e1d34c1edc306a6959` |
| Isolated Mini checkout | `/Users/n/MetalRobo-numilab-spatial-58dc262` |
| Device | Apple M4 Pro on `macmini` |
| Source program package | 10 canonical 2,512-byte programs and 35 canonical 64-byte state inputs |
| GPU-probe SHA-256 | `c9d246e42b6d1a2f8b129337602bdf3d40cd2d9450ca8367894239aef3bdf176` |
| Metal-library SHA-256 | `fdeca34b9ccb492fd874dc3879e2b77434b97a79c79e3c834494a0f0c5619ecd` |

The isolated Mini checkout was configured with AppleClang `21.0.0.21000101`.
The Function, spatial-transform, and GPU probes passed. Every one of the 35
source program/input cases then ran the device transform followed in the same
command buffer by `H`-transpose projection of a fixed finite source-frame
wrench and `Hdot*qdot` spatial-bias evaluation. Each result matched the
decoded FP64 Core evaluation within FP32 tolerance, and both full GPU payloads
were byte-identical on the repeated invocation.

The fixed wrench is a numerical projection probe, not a Rajagopal muscle,
contact, or measured load. The current Core still does not assemble these
columns into a multi-body mass/bias solve or advance a FunctionBased body, so
the CustomJoint skeleton gate remains blocked.

## Canonical Rajagopal CustomJoint program sidecars — 2026-08-26

This is source-derived kinematic evidence for every Rajagopal `CustomJoint`.
It is not articulated dynamics, contact, muscle actuation, or full-human
evidence.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Numi runtime revision | `bfd95d22413290215cdeba55ad51d8abea5f2d33` |
| Isolated Mini checkout | `/Users/n/MetalRobo-numilab-spatial-bfd95d2` |
| Device | Apple M4 Pro on `macmini` |
| Source program package | 10 canonical 2,512-byte programs and 35 canonical 64-byte state inputs |
| Sample source-program SHA-256 | `walker_knee_r.mrospatial`: `834fc89a5efdc999b23a34d55bc1e7bf8782a90479f4f4deedce337ac7da0ebf` |
| GPU-probe SHA-256 | `de96c25362aae09d85fd31b960bf79e380e2c4c99beac65b1d65a0a5a389bbf7` |
| Metal-library SHA-256 | `1853cf21ee01fdee89466d87df054f612ce1cee5a25b690aa4ceb27d0c90facb` |

The isolated Mini checkout was configured with AppleClang `21.0.0.21000101`.
`metalrobo_opensim_function_probe` and
`metalrobo_opensim_spatial_transform_probe` passed. The device run then
loaded each generated program and its default plus each coordinate's
unit-velocity sidecar: 35 source-artifact GPU cases in total. Each case
validated the fixed binary with a Core decode/re-pack byte round trip,
compared source-order pose, `H`, and `Hdot` against the decoded FP64
evaluator, and compared two GPU payloads byte-for-byte.

The Core program is not admitted into ABA, does not project generalized
forces, and does not step a body. Therefore the CustomJoint skeleton gate
remains blocked despite this complete bounded kinematic evidence.

## OpenSim FunctionBased spatial-transform program — 2026-08-26

This is source-derived kinematic evidence for the pinned Rajagopal
`walker_knee_r` `CustomJoint`; it is not articulated dynamics, contact,
muscle actuation, or full-human evidence.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Numi runtime revision | `a1b81d168b79b117c1b81d145d0838a571aeeb4e` |
| Isolated Mini checkout | `/Users/n/MetalRobo-numilab-spatial-a1b81d1` |
| Device | Apple M4 Pro on `macmini` |
| Source test state | `knee_angle_r = 0.43 rad`, `qdot = -0.71 rad/s` |
| GPU-probe SHA-256 | `980ec46741b6251b51c53f958e19039f0baaaa25b09fbc4f6fe5f74d88f43d9d` |
| Metal-library SHA-256 | `a0c8968b61d9f5183e8dc99e631f63b8b14f707a2549e263dec0868768f5e081` |

The isolated Mini checkout was configured with AppleClang `21.0.0.21000101`.
`metalrobo_opensim_function_probe` and
`metalrobo_opensim_spatial_transform_probe` both passed. The GPU probe then
ran twice and each invocation reported:

```text
opensim_spatial_transform_gpu=ok device=Apple M4 Pro \
  tx=0.000271367 h_angular_x=0.993748 hdot_linear_x=-0.000971865
```

The probe packs the source-order six-axis FunctionBased program, evaluates the
pose, `H`, and `Hdot` on device, compares every returned component with the
compiled CPU program within FP32 tolerance, and compares the two GPU payloads
byte-for-byte. It does not feed the program into ABA, project forces, or step
an articulated body; the CustomJoint skeleton gate remains blocked.

## Rajagopal distal-leg PinJoint preview — 2026-08-26

This is evidence for a deliberately reduced source-derived preview. It is not
evidence for a complete NumiLab Human, collision/contact, muscle actuation,
material behavior, or physiological accuracy.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Numi runtime revision | `bafc28d081ec4b085aaaf663869c0b07476bec29` |
| Device | Apple M4 Pro on `macmini` |
| Probe SHA-256 | `9c5b10d5d2bf39d4a8975707b2fc6b7c421dc497e25b9198982cfabbdbadaa19` |
| Runtime library SHA-256 | `6204cc5554daa3375977c3b93c193c2d025d20275b503f617172c6d4b1974b13` |
| Metal library SHA-256 | `3c44ddf5c18f6a58a6ff96240d26084a19f2c2450bcb0ea0457be1251acd418d` |

The Mini did not have CMake installed. The probe, runtime library, and Metal
library were therefore built locally at the listed runtime revision, copied to
an isolated Mini worktree at that same revision, and SHA-256 matched before
execution.

The command was run twice per side with zero generalized effort:

```sh
DYLD_LIBRARY_PATH=$PWD \
  ./metalrobo_robot_description_cooker_probe \
  --metal rajagopal-right-distal-pin-preview.urdf
```

| Side | URDF SHA-256 | Compiled model | Repeated state fingerprint |
| --- | --- | --- | --- |
| Right | `10692ea7732c4daa582d45587c55c9fb970e6f19f9a750929b12f5ee15a3620e` | 4 links, 3 revolute joints, 9 DoFs, 0 colliders | `3066262497575343521` |
| Left | `4e66db215427bfdbb682aec1fc04dc4e83483a26b5bd26ca3c306216bd1f950d` | 4 links, 3 revolute joints, 9 DoFs, 0 colliders | `4775400414694806446` |

Each of the four submissions reported `robot_description_external_metal=ok`,
`gpu_status=0`, and its side's identical fingerprint on the repeated run. The
fingerprint covers the returned acceleration, next-velocity, next-position,
and GPU-status payload on this exact binary/model/platform combination; it is
not a cross-platform serialization or a performance metric.

The preview contains tibia, talus, calcaneus, and toes joined by exact source
ankle, subtalar, and MTP PinJoint transforms. It intentionally contains no
BodyParts3D collision mesh, no OpenSim muscle or tendon lowering, no contact,
and no deformable anatomy.
