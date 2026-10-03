# Patellofemoral pose intersection audit — 2026-10-03

The neutral-coordinate rebase fixed the patella's front/back position at literal
`qpos0`, and every patella vertex stayed anterior to its knee-anchor plane in
the eight previously sampled poses. That did not establish that the patella
and femur surfaces remained separate as the knee flexed.

The [pose-intersection receipt](media/numi-human-patellofemoral-pose-intersections-20261003/receipt.json)
checks both the pinned MyoSim source bone geoms and the current registered
NHBONES1 ABI 3 meshes after the MyoSim equality projection. It checks the eight
existing source poses and samples the full declared knee-angle range from 0 to
2.0944 rad in 0.05 rad increments plus the exact upper endpoint. Binary64 world
coordinates are lifted exactly onto a common integer lattice for the triangle
intersection predicates. Samples whose projected coordinates leave the
declared source or native joint ranges are retained as diagnostics and excluded
from the range-valid gate. The receipt also rechecks the consumed rigid and
equality programs against the pinned source; both source-program checks pass.

The source-range sweep contains 43 samples; 34 pass all projected range
checks. In those valid samples, 41 side-pose states contain a total of 1,631
exact triangle-pair intersections in the registered meshes. The pinned MyoSim
source geoms also intersect in 56 side-pose samples, with 1,236 triangle-pair
intersections. The eight-pose suite catches source-geometry crossings in both
knees at 0.9 rad flexion, 1.4 rad deep crouch, and 0.7 rad functional crouch;
the BodyParts3D registered surfaces cross in the 0.9 and 0.7 rad poses. The
result is `failed_sampled_surface_intersection_gate`.

The crossings are present in the pinned MyoSim mesh pair, so changing only the
BodyParts3D registration cannot resolve this source tracking gap. This is not an
expert clinical verdict, contact-pressure result, or proof that a particular
real person's patella tracks incorrectly. Removing these crossings safely
requires a source-bound patellar path that avoids the femur while retaining
QAT/PTL/cartilage owners and respecting source/native range constraints; an
uncalibrated shift would not close that mechanical problem. The audit is
sampled rather than continuous and does not test closed-volume containment,
cartilage contact, loaded tracking, clinical calibration, or runtime
acceptance.

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
