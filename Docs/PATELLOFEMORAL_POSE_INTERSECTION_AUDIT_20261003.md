# Patellofemoral pose intersection audit — 2026-10-03

The neutral-coordinate rebase fixed the patella's front/back position at literal
`qpos0`, and every patella vertex stayed anterior to its knee-anchor plane in
the eight previously sampled poses. That did not establish that the patella
and femur surfaces remained separate as the knee flexed.

The new [pose-intersection receipt](media/numi-human-patellofemoral-pose-intersections-20261003/receipt.json)
tests the current registered patella and femur meshes from NHBONES1 ABI 3 after
the pinned MyoSim equality projection. It checks the eight existing source
poses and samples the full declared knee-angle range from 0 to 2.0944 rad in
0.05 rad increments plus the exact upper endpoint. Binary64 world coordinates
are lifted exactly onto a common integer lattice for the triangle intersection
predicates. Samples whose projected coordinates leave the declared source or
native joint ranges are retained as diagnostics and excluded from the
range-valid gate. The receipt also rechecks the consumed rigid and equality
programs against the pinned source; both source-program checks pass.

The source-range sweep contains 43 samples; 34 pass all projected range
checks. In those valid samples, 41 side-pose states contain a total of 1,631
exact triangle-pair intersections. The existing eight-pose suite has four
affected side-pose states: both knees at 0.9 rad flexion and both knees in the
0.7 rad functional crouch. The result is
`failed_sampled_surface_intersection_gate`.

This is a geometry failure for this registration and prescribed source pose
law. It is not an expert clinical verdict, contact-pressure result, or proof
that a particular real person's patella tracks incorrectly. Removing these
crossings safely requires resolving the source patellar tracking path against
the registered femur while retaining QAT/PTL/cartilage owners and validating
the same source and native range constraints; applying a cosmetic translation
would not close that mechanical problem. The audit is sampled rather than
continuous and does not test closed-volume containment, cartilage contact,
loaded tracking, clinical calibration, or runtime acceptance.

Reproduce through the owner CLI with the pinned MyoSim environment:

```sh
numi human myosim-patellofemoral-pose-intersection-audit \
  --sources Sources \
  --artifact Docs/media/numi-human-patella-neutral-rebase-geometry-20261003/core-artifact \
  --bone-artifact Docs/media/numi-human-patella-neutral-rebase-geometry-20261003/bone-artifact \
  --registration Docs/media/numi-human-patella-neutral-rebase-geometry-20261003/lower-limb-registration.json \
  --output <new-receipt-path> \
  --python /private/tmp/numi-human-myosim-20261003/bin/python
```
