# Organ-to-blood cavity bridge, 13 September 2026

The bridge receipt at
`Docs/media/organ-blood-cavity-bridge-20260913/bridge.json` binds the four
CVSim21 cardiac chamber references to the exact BodyParts3D cavity members in
the current organ source-moment receipt.  It verifies the archive member hash,
semantic chamber identity, source-frame centroid, first moment, and central
second moment for each member.  The hydraulic initial and reference volumes are
reported beside the source surface-integral volume so their disagreement is
visible instead of being hidden by a scale fit.

The four source members and initial hydraulic-to-source-integral ratios are:

| Chamber | Member | Source integral (m³) | Initial ratio | Reference ratio |
| --- | --- | ---: | ---: | ---: |
| right atrium | `FJ2424` | 8.45515320178338e-05 | 0.38158918844195644 | 0.165579495319459 |
| right ventricle | `FJ2423` | 1.1697235641012262e-04 | 1.1828319764825057 | 0.39325530759350613 |
| left atrium | `FJ2425` | 5.193679901140525e-05 | 0.9191457727131351 | 0.4621000996755621 |
| left ventricle | `FJ2422` | 9.793877336968795e-05 | 1.6209652363743112 | 0.5615753404669708 |

The source-moment receipt is SHA-256
`4d4e1cbdfa650a0d3fc4b25ee2f0bdaf13e45a3278189f5b7328b47bf1762ea2`.  The
bridge identity is
`b639799a498e56c787e54f7f96c2d28181dd1099798d58e283fae7f530d75611`; the
current CVSim source-native identity is
`eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818` and the
source cavity geometry identity is
`849bd41e2fa7e172e8cda2c6889658eda203c4c5e240b3a2adb7c5649c9243c1`.

The compiler is available as
`numi organ-blood-cavity-bridge --output <new-receipt.json>`.  The independent
verifier reports four bindings and 42 intersecting triangle pairs.  Hydraulic
CVSim volumes remain the only volume authority.  The source surface integral is
not admitted as a physical volume, and no density, mechanical blood owner,
body-frame transform, pressure-gradient momentum transfer, two-way tissue
coupling, material calibration, or standing/walking claim is made.  The bridge
therefore closes source identity and comparison bookkeeping while preserving
the anatomical and physiological gates that still require registration,
materials, loading, and held-out calibration.

The separate [cardiac cavity body-link receipt](ORGAN_CARDIAC_CAVITY_BODY_LINK_20260913.md)
binds these four source members to the exact MyoSim torso source/core frame.
It is frame bookkeeping only and leaves physical-volume, blood-mass, material,
pressure, tissue-exchange, and disjoint-domain ownership unresolved.

As a native regression check after authoring the bridge, the existing ABI38
owner was rerun on the physical Apple M4 Pro from isolated branch
`human-blood-mass-20260913` at `f89991dc55c591fb6dc5ca5388de4b9034592c5a`.
The compiler, vascular runtime, and cavity owner checks passed; the retained
log and SHA-256 are in
`Docs/media/organ-blood-cavity-bridge-20260913/native/owner-recheck.log`.

## Current native requalification

The current native tree was requalified on the physical Apple M4 Pro from
branch `human-blood-mass-20260913` at commit
`e07026ab3ad497a869ee247cdfd1aa23ecc16ebb` (tree
`4a9b0fe7b62f4809b9621a0f175ba06a20aa952a`).  The focused CTest selection
passed all 17 of 17 tests.  It covers the vascular compiler and Metal path,
Human physiology admission and payload, CVSim source and admission, cardiac
source and transaction, regional material/frame/reference checks, synthetic
Human binding and cavity checks, and tissue mass partition.  The complete
CTest output is retained in
[`native-organ-blood-ctest.log`](media/organ-blood-cavity-bridge-20260913/native-current/native-organ-blood-ctest.log),
with the identity, receipt, and per-file hashes in
[`native-current/`](media/organ-blood-cavity-bridge-20260913/native-current/).

The direct cavity-owner output independently records `blood_mass_owner=pass`,
`partitioned_inertia=pass`, `dynamic_spatial_moments=pass`,
`time_integrated_moment_closure=pass`, `pressure_gradient_reaction=pass`,
`pressure_driven_fluid_momentum=pass`, bitwise replay, and Jacobian checks on
the same Apple M4 Pro.  It also explicitly reports
`pressure_driven_momentum=unqualified`, `anatomical_registration=unqualified`,
`subject_calibration=unqualified`, and absent biological calibration for the
synthetic moving-wall interface.  That boundary is retained in
[`vascular-cavity-owner.log`](media/organ-blood-cavity-bridge-20260913/native-current/vascular-cavity-owner.log):
the native result qualifies the engineering contracts and synthetic cavity,
not body-frame anatomy, subject-specific density/materials, two-way tissue
exchange, physiology, or standing/walking.

After the native behavior-clock alignment, the same physical M4 Pro selection
was rerun from commit `ac3b4af73c6ae8efef6f07d6e626e71b52e68c75` (tree
`4d98915adff9a43964a7e77088212f52953966e7`) and again passed 17/17. The
post-clock log and receipt are retained in
[`native-current-post-clock/`](media/organ-blood-cavity-bridge-20260913/native-current-post-clock/)
with SHA-256 manifest; the qualification boundary is unchanged.

The next native increment is recorded in [`FULLBODY_VASCULAR_ADMISSION_20260913.md`](FULLBODY_VASCULAR_ADMISSION_20260913.md). On the physical Apple M4 Pro at commit `8d87e9d46a1f973a6fbd906b90899c2799fe4f6e` (tree `91fc1fc72fb0320c7a4bc6e01a22073ad5162807`), the real 157-body/129-q/128-v fullbody package admitted one explicit synthetic vascular owner: two compartments, one inertial connection, one normalized four-node FEM tissue region, density `1060 kg/m^3`, explicit momentum-transfer metadata, and bitwise package replay. The focused fullbody CTest selection passed 3/3, with the receipt and logs in [`native-fullbody-vascular-admission/`](media/organ-blood-cavity-bridge-20260913/native-fullbody-vascular-admission/).

The follow-up [coupled-support requalification](COUPLED_SUPPORT_REQUALIFICATION_20260913.md) at native commit `cfef55a8199c2167fbe9beb04417a47db705db5d` (tree `a842948af74698501546c46de98eb7b36277e528`) repairs the invalid-support publication boundary. At synthetic fixture pressure `0.0008` and the exact 12,500 ns clock, the package reaches a successful coupled root and all ten support receptor rows are valid. A deliberately higher `0.001` pressure remains a retained fail-closed case: Matter status 6 is returned and no accepted Human token is published. This adds dynamic support/root engineering evidence without promoting the result to anatomical registration, prescribed supports/loading, activation or subject calibration, unresolved materials, force convergence over a calibrated range, two-way tissue exchange, physiology, or standing/walking.
