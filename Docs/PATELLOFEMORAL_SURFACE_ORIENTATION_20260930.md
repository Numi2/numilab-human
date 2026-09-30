# Patellofemoral surface orientation - 30 September 2026

The exact Open Knee(s) `oks003` patellar cartilage (`PTC`) has its contact
surface on the posterior, femur-facing side and its bone tie on the anterior
side. The compiler now rejects a contact or tie triangle that is degenerate,
internal, or wound toward its owning tetrahedron. It applies the same check
after registration and after the right-knee mirror parity correction; it also
rejects reversal of the two patellar cartilage layers. The output payload
format and geometry are unchanged.

The [independent source and payload audit](media/patellofemoral-surface-20260930/receipt.json)
decoded both `NHKNEE1` payloads and matched their selected triangle and
tetrahedron indices to the pinned source. Every selected face is a unique
exterior face with an outward normal: **11,053** patellar cartilage contact,
**8,875** patellar cartilage bone tie, **22,478** femoral cartilage contact,
and **18,346** femoral cartilage bone tie faces, on each side. The patellar
contact surface's area-weighted anterior cosine is **-0.8095** and its bone
tie's is **+0.8768** in the compiled coordinates. The right side is the
declared sagittal mirror of the left source, not an independent segmentation.

Among patellar contact triangle centroids whose nearest femoral contact
triangle centroid is within 1 mm, there are **1,233** faces on either
compiled side, with a **0.443 mm** median centroid separation. **96.9%** of
those nearest face-normal pairs have a dot product below -0.5; **99.2%** of
the patellar normals and **99.8%** of the corresponding femoral normals point
toward the other centroid. This establishes the facing of a *nearby static
patch*. Centroid separation is not the minimum surface gap, and most of the
authored patellar contact faces are outside that 1 mm patch. The femoral
contact set is a broad exposed surface reused by other source contact pairs.

These checks close the static posterior/anterior articular-face winding gap
for the pinned source and its two compiled placements. They do not verify an
exact cartilage gap, contact law or pressure, load transfer, sustained loaded
flexion, clinical anatomy, or the historical standing video's anatomy. The
five authored dependent-coordinate range conflicts in the separate lower-limb
pose audit remain open.

The later [exact contact-surface audit](PATELLOFEMORAL_CONTACT_INTERSECTIONS_20260930.md)
finds 18 source-authored crossing triangle pairs on both compiled sides.
Correct outward winding therefore does not imply static noninterpenetration.

Reproduce the audit after compiling both sides with the Open Knee compiler:

```sh
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/verify_patellofemoral_surface.py
```
