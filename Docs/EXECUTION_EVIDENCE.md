# Bounded execution evidence

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
