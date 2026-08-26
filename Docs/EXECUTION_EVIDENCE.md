# Bounded execution evidence

## FunctionBased device-updated Millard excitation streamed source-contact smoke — 2026-08-26

Core revision `bb6f8f26275cc132f4cddcf77dc461addbbe18d9` adds a device
activation-control stage to the bounded FunctionBased streamed temporal-cone
response path. A local Apple M4 run loaded the complete 22-body Rajagopal
source tree and 80-muscle ABI-v3 payload, added only a synthetic static plane
and source-body sphere, then completed two direct-effort contact steps with one
active contact and one constraint. The source-default submission published
muscle force L1 `37802.936219`. A second submission supplied the packed
`[control step][environment][muscle]` stream with every excitation at one; the
device applied the exact first-order activation hold with explicit `0.01 s` and
`0.04 s` time constants before its own force projection and published
`60180.837369` L1. Those explicit values are a smoke-test input matching the
separately recorded OpenSim Millard class defaults, not a calibration claim.
The same probe still passed the three-step source-dynamics parity gate: maximum
acceleration error `1.75e-04`, velocity error `1e-06`, and configuration error
`0` at printed precision.

| Item | SHA-256 |
| --- | --- |
| Rigid source payload | `da7e52ddd64728ed0a63e73a11cf857ec5489b3eb29e32d11f352f35507cdee6` |
| Millard source payload | `ecc900d71369c3c0cbf7a09fbdc33a2194f6a77edcddf594d306d850de60fbf4` |
| Probe binary | `4bed5319f52fe76983a07254453edadb94e3ef41f42ac6e722af3c2817827cc6` |
| Core library | `b04703e4733fef9ce97b471d966cad45d412574a86587fb1649f93e412beb676` |
| Metal library | `941b8eb24fa00d6a2431d80285f9bb430a781c28a601d6e36c3c12db686cb6d1` |

This is a local device smoke test, not the pending Mac mini reproduction or a
walking result. The plane/sphere are test-only collider proxies, not
BodyParts3D registrations. The bounded path is fixed-root and direct-effort;
no mobile-root gait policy, anatomical contact material, deformable anatomy, or
visual motion is qualified by it.

## FunctionBased inverse-response reference — 2026-08-26

Core revision `38cb6bb` adds the FP64 `M⁻¹·rhs` oracle used to qualify future
FunctionBased contact-response columns. On the M4 Pro it loaded the complete
22-body, 35-DoF Rajagopal tree and recovered a unit generalized impulse through
the source mass matrix within `1e-10`. Probe SHA-256:
`6be91d6093cf32524acd708cfff9a1bb0eeb64ab7be59f0e2402dff1950c4087`.

This is an inverse-response *reference* only. Metal contact response columns,
registered feet, contact parameters, policy rollouts, and walking remain open.

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

## Bounded FunctionBased skeleton and active Millard actuation — 2026-08-26

Core revision `14c64f3` was built in the isolated Mini checkout and ran the
source-locked Rajagopal reference on the Apple M4 Pro. This is the current
mechanics qualification; older sections that describe an operator-only or
static-sidecar boundary are historical evidence for preceding revisions.

| Item | Exact value |
| --- | --- |
| Core revision | `14c64f3` (`origin/coupled`) |
| Source rigid payload SHA-256 | `da7e52ddd64728ed0a63e73a11cf857ec5489b3eb29e32d11f352f35507cdee6` |
| Source Millard payload SHA-256 | `ecc900d71369c3c0cbf7a09fbdc33a2194f6a77edcddf594d306d850de60fbf4` |
| Device / toolchain | Apple M4 Pro on `macmini`; AppleClang `21.0.0.21000101` |
| Probe SHA-256 | `f5a85f3e4f4e98a4171b38f19e0dc750c18241e4e3ed41018d4a35318844a560` |
| Core library SHA-256 | `08d548091af84d460f1c326cf3b7cc6b67fa89b2fbd266ed9b585d1a69a0d59d` |
| Metal library SHA-256 | `8351863cbbf5ce523956d9b49484ae39c315f9b8bdec54285544bcffaff71922` |
| Device log SHA-256 | `ed9ffd6d83df248f8cb0ed8d965a507066fe5ed23dafb998936aeec3b7715269` |

The run admitted one fixed-root source tree with 22 source bodies, 35
coordinates/velocities, and 10 FunctionBased programs to MetalWorld's
resident free-motion state step. Three persisted direct-effort steps matched
the FP64 reference with maximum acceleration, velocity, and configuration
errors of `1.75e-04`, `1e-06`, and `0` at printed precision.

The Millard payload is ABI v3: all 46 source `PathWrap` methods and 1-based
ranges are carried into both Core paths. All source methods are `hybrid`; four
authored ranges are `[2,3]` and were no longer treated as unconstrained. The
same command buffer then evaluated all 80 source Millard muscles from the
private FunctionBased pose/Jacobian streams, applied 26 finite-cylinder wraps,
and reduced their per-muscle force vectors into MetalWorld's working-effort
arena before the source-dynamics step. The aggregate device force L1 was
`12065.227945`; its relative difference from the CPU source bridge was
`3e-06`. The force changed accepted velocity from the passive state by
`89.405083`, and aggregate acceleration L1 relative error was `8e-06`.

This qualifies bounded source skeleton and static-equilibrium muscle actuation
on device. It does not authenticate the unavailable MoBL-ARMS archive,
establish OpenSim binary or hybrid-wrap-history equivalence, register
BodyParts3D frames/colliders, qualify contact or deformable anatomy, or supply
tissue material calibration.

## Device-resident Rajagopal Millard reference pass — 2026-08-26

This is a bounded device execution of a static Millard active-force reference.
It is not Metal ABA or MetalWorld state stepping, a persistent muscle-state
integrator, contact/deformable-anatomy evidence, an OpenSim binary-equivalence
result, or a material-calibration result.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Core revision | `4e1f4e9d95f3632826cf60b72b2ab9cef394c612` |
| Isolated Mini checkout | `/Users/n/MetalRobo-numilab-human-functionbased-837ccb2` |
| Device / toolchain | Apple M4 Pro on `macmini`; AppleClang `21.0.0.21000101` |
| Millard payload | ABI v2; 80 muscles; 288 source path points; 46 source path wraps; 22 source curve scalars per muscle |
| Millard payload SHA-256 | `101b2a549e4d20145391138e268642e6cd2ab99bd77005ad3f6ab5875b3a08c1` |
| Device probe SHA-256 | `d876b551bb07d8117f4dee51db76e000c5860dc252c0d55bc24de6c7b6115c1e` |
| Metal library SHA-256 | `b2085a46b0e4b804c73114719751335095201e7520228842ea96977506e86a56` |
| Device run log SHA-256 | `c84f1c71ad23ee43bbd7cd2eb8e5d8b4ffa400422b102a5130cca8378357ebe1` |

The generic FunctionBased operator first produced source-tree body poses,
path-point world positions, and analytic point Jacobians in its private Metal
buffers. The typed Millard pass then consumed those buffers in the *same command
buffer*, reconstructed the source-materialized curves, applied finite-cylinder
wrap selection, solved static fiber-tendon equilibrium, and wrote one
generalized-force vector per muscle. All 80 muscles completed; the device and
FP64 bridge selected the same 26 active cylinders. Maximum per-muscle relative
errors were `0` for path length at printed precision, `3.11e-04` for tendon
force, and `1.70e-04` for generalized-force L1; the maximum device equilibrium
residual was `4.899e-03` N.

This closes the source-backed device-reference gate for the imported pose. It
does not lower muscles into MetalWorld, advance activation/fiber/tendon state,
admit the FunctionBased skeleton to Metal ABA, or validate the source against
an OpenSim binary/trajectory or measured anatomical moment arms.

## Rajagopal Millard static-equilibrium reference and combined device gate — 2026-08-26

This is a source-code-level Millard reference and a separate whole-tree Metal
operator invocation. It is not a device-resident muscle actuator, Metal ABA or
MetalWorld state step, OpenSim binary-equivalence result, contact run, or
material calibration.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Core revision | `479ccdf905c3b0145e13cca9c673b410ecea5b4f` |
| Isolated Mini checkout | `/Users/n/MetalRobo-numilab-human-functionbased-837ccb2` |
| Device / toolchain | Apple M4 Pro on `macmini`; AppleClang `21.0.0.21000101` |
| Millard payload | ABI v2; 80 muscles; 288 path points; 46 source path wraps; 22 source curve scalars per muscle |
| Millard payload SHA-256 | `101b2a549e4d20145391138e268642e6cd2ab99bd77005ad3f6ab5875b3a08c1` |
| Combined-reference probe SHA-256 | `3e3f66c0e10b2e589b424c90363975540d73ec9c56d14fa67cc6dfe3d91bacca` |
| Millard reference-probe SHA-256 | `feb17da21ccd009b2a32119ca413efa36b4c9704cd730ce301e9a4bb8e8e3de5` |
| Metal library SHA-256 | `9dbb88b2adeb2e3d7456dcddc7411c0ecb83c6bebb58ca976608c4768f4ce8c3` |
| Combined run log SHA-256 | `3b25bd0eda222a37a45bdabeaa56878f3f1a5b3d3474f5e5beafde138e9f0c46` |

The Mini rebuilt the exact Core revision, ran the generic finite-cylinder
reference probe, then loaded the matching full Rajagopal rigid and Millard
payloads. All 80 source muscle records reached static fiber-tendon equilibrium
at their default activation and the imported reference pose; all 288 source
attachment points and 46 source wrap records were admitted. Twenty-six
cylinders were geometrically active in that one pose. The resulting tension
projection had generalized-force L1 norm `1.653391e+04` and maximum reported
equilibrium residual `0` at the probe precision.

The same invocation ran the pre-existing whole-tree FunctionBased Metal
operator: body poses, point Jacobians, and the 35x35 mass matrix agreed with
the FP64 reference within its existing FP32 thresholds (absolute mass error
`1.7e-05`, scaled `2e-06`; printed pose and point errors `0`). The muscle
reference itself runs in the owner FP64 Core path. Its source curve construction
and finite-cylinder projection still require a pinned OpenSim comparison,
including hybrid wrap-history behavior and measured moment arms, before any
OpenSim-equivalence claim.

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

## Whole-tree FunctionBased Metal articulated-operator gate — 2026-08-26

This is device execution of the bounded articulated *operator*, not Metal ABA
or MetalWorld state stepping, BodyParts3D registration, contact, or muscle
actuation.

| Item | Exact value |
| --- | --- |
| Rajagopal source SHA-256 | `8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec` |
| Core revision | `380f96bd8baf691980197aebd162a0f9d19c5aa7` |
| Isolated Mini checkout | `/Users/n/MetalRobo-numilab-human-functionbased-837ccb2` |
| Device / toolchain | Apple M4 Pro on `macmini`; AppleClang `21.0.0.21000101` |
| Payload SHA-256 | `da7e52ddd64728ed0a63e73a11cf857ec5489b3eb29e32d11f352f35507cdee6` |
| Probe binary SHA-256 | `ca0293d14f06a886cafdacbf46ab6e6991083866f1dda506f934ef6732d158db` |
| Metal library SHA-256 | `9dbb88b2adeb2e3d7456dcddc7411c0ecb83c6bebb58ca976608c4768f4ce8c3` |
| Operator log SHA-256 | `8674feda46783794ee092379485f637f29f8ed053b72bc218ac78a413628f297` |

The complete source tree (22 source bodies; 23 engine bodies including its
fixed synthetic root; 22 joints; 35 `q`/`v`; and 10 FunctionBased programs)
passed one Apple-GPU operator invocation at a deliberately non-neutral
configuration. It verified body poses and non-collinear point Jacobians against
the FP64 reference, then assembled and factorized the 35×35 mass matrix. The
maximum reported FP32-to-FP64 mass error was `1.7e-05` (scaled `2e-06`); all
reported pose and point-Jacobian errors rounded to zero at the probe's printed
precision. This qualifies only the owner Metal operator path. It does not
qualify ABA/MetalWorld state advancement, contact, muscles, anatomy
registration, OpenSim trajectory equivalence, or material behavior.

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
