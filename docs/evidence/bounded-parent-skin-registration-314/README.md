# Complete bounded parent-cell queries for skin source registration

The registered O/I/D exporter previously queried every same-material parent
tetrahedron in a sphere sized by the largest cell in the whole body. Its exact
scalar projection then ran on many distant cells.

The same parent cells now share size-binned centroid-sphere indices and AABBs.
Triangle-inequality sphere bounds retain every possibly nearer cell and every
tie. Point-to-box lower bounds reject provably distant cells before calling the
unchanged scalar closest-point evaluator. No fixed nearest-K truncation,
tolerance, source identity, material, frame or tie-rejection policy changes.

Validation:
- Existing full-86-binding, affine-registration and ancestry-negative fixtures.
- Thirty exhaustive heterogeneous-cell queries, coincident ties and distinct
  projection rejection; small and large cells share the same material.
- Three phase-alternating matched pairs over 24 actual O286 skin queries:
  all returned fields bit-for-bit identical. Median sample speedup 145.70x
  (legacy about 2.91 seconds; pruned about 0.020 seconds).
- Full O286 preparation still rejects vertex120 at the existing default tie
  guard. Legacy rejects identically; exact witnesses are retained here.

These timings cover CPU asset preparation only, not a complete registration,
GPU solver, native physical admission or motion qualification. The final
physical skin/fat bundle remains unqualified.
