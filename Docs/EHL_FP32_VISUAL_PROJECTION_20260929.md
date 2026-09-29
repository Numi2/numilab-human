# Right EHL compiled visual-face repair

The prior [muscle-surface census](MUSCLE_SURFACE_EMBEDDEDNESS_20260929.md)
found an exact zero-area triangle in the emitted right extensor hallucis
longus (`FJ1408`) surface. Rechecking all 1,480 faces found two zero-area
triangles, IDs 1475 and 1477. Their authored BodyParts3D vertices are
distinct. The visual attachment projection put two vertices at the same
compiled Float32 position near the named distal hallux phalanx. The left EHL
did not have this collapse.

The importer now checks the projected surface after actual NHTISS4 Float32
packing. If a face collapses, it searches powers-of-two steps back toward
the unchanged source points and admits the first projection with no exact
zero-area face. The search stops if the retreat would exceed 10 micrometres;
an unresolved source fails import rather than silently emitting a degenerate
face. On this pinned source, it chose a `2^-19` backoff. The maximum world
retreat from the visual target was `1.5990718227521527e-8 m`.

The [payload delta receipt](media/ehl-fp32-projection-20260929/receipt-v1.json)
compares the prior and repaired full 150-surface NHTISS4 packages. Only stable
surface 23 changed: 71 vertex positions and 83 normals, with a maximum
compiled position delta of `2.006130221163669e-8 m`. Every face index, body
binding, sparse skinning index and weight stayed byte-identical. The
[new exact census](media/muscle-surface-embeddedness-20260929/receipt-v2.json)
checks all emitted surfaces again.

The right EHL can now run the exact intersection predicate, but remains
**open and self-intersecting**: 186 boundary edges and three intersection
pairs. This corrects an emitted Float32 defect, not the source's open
topology, the visual attachment's clinical placement, tendon-to-bone force
transfer, or sustained standing. The unprojected MyoSim route and its
mechanics are unchanged.
