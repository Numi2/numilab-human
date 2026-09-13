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

As a native regression check after authoring the bridge, the existing ABI38
owner was rerun on the physical Apple M4 Pro from isolated branch
`human-blood-mass-20260913` at `f89991dc55c591fb6dc5ca5388de4b9034592c5a`.
The compiler, vascular runtime, and cavity owner checks passed; the retained
log and SHA-256 are in
`Docs/media/organ-blood-cavity-bridge-20260913/native/owner-recheck.log`.
