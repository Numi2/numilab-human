# Bounded execution evidence

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
