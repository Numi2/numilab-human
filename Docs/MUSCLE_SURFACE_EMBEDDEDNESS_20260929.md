# Compiled muscle and tendon surface embeddedness

The earlier muscle geometry audit classified the original BodyParts3D OBJ
members. It counted 60 single-closed muscle source surfaces, but did not test
self-intersections or the geometry actually emitted to NHTISS4. The new
[baseline receipt](media/muscle-surface-embeddedness-20260929/receipt-baseline.json)
tests the exact-coordinate quotient of the compiled Float32 payload: only 34
of those 60 are single, closed, and free of exact self-intersections. The
60-member algebraic-volume and route-incidence receipts remain historical
source-geometry bookkeeping, not an admission of 60 embedded compiled volumes.

The importer now cancels only exact coincident opposite-winding face pairs
before computing normals, bindings, and indices. The pinned 150-surface build
cancelled 462 pairs on 80 surfaces. This removes 462 unique visual triangles,
with 2.7145019056868547 mm² total source-space area; it is a measured loss
of rendered support, not a claim that the visual mesh stayed identical. No
points were moved and no replacement faces or caps were added. The
[repaired-payload receipt](media/muscle-surface-embeddedness-20260929/receipt-v1.json)
verifies that every retained compiled Float32 triangle has exactly the same
three positions as the baseline, in the same order. All 434,917 compiled
vertex records, including normals and sparse motion weights, are byte-identical
to the baseline. Both payloads have the same NHTISS4 ABI, source identity,
registration fingerprint, 150 surface IDs, and 512 body bindings.

The repaired build admits 52 of 148 muscle surfaces as *single embedded
surface candidates*: the original 34 plus 18 whose source topology was
defective. No tendon surface passes that gate. Of all 150 emitted surfaces,
82 are closed and individually embedded but 30 of those have multiple
components; 62 are closed but self-intersecting, three remain open or
nonmanifold and intersecting, two remain open or nonmanifold without detected
intersections, and one has an exact degenerate triangle. Thus the existing
60-source-volume candidate cannot be promoted wholesale to an embedded
compiled-tissue claim. The 18 newly admitted surfaces have no new volume
moment, density, mass, material, or active-force owner.

These are rest-configuration, per-surface geometric checks. Cross-surface
overlap and containment, anatomical placement under motion, clinical anatomy,
skin and tendon mechanics, organ interfaces, physical volume/mass ownership,
and standing qualification remain open. In particular, this receipt does not
establish that the repaired muscles attach to the clinically correct sites.
