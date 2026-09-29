# Numi Human right-eye source components — 29 September 2026

The source-rest full-support laterality audit found three right-eye source
members still carrying small fragments on the left side of the body's
sagittal midline. The right cornea (FJ1340, source ID 413) had 18 such
triangles; the right sclera (FJ1368, ID 426) had four; the suspensory ligament
of the right lens (FJ1371, ID 427) had four. All three area-weighted centroids
were on the named right side, so centroid checks did not detect this defect.

Each excluded triangle belongs to a complete edge-disconnected component
whose vertices and exact Float32 positions are disjoint from retained eye
support. The source-face exclusions are **4588–4605**, **39520–39523**, and
**7372–7375**, respectively. The affected source members and every preceding
surface remain byte-identical in a new ABI 5 packet. New inspection copies
600, 601 and 602 retain every other face and vertex at its exact compiled
position; they add no face or anatomical member. Source ID 412 continues to
use the earlier corrected inspection copy 599.

The resulting native source-rest packet contains all **602 surfaces**. Its
body poses match the prior 599-surface packet, and the independent native
audit checks every surface's owner, stable ID, local geometry bytes, instance
range and visibility. The full-support laterality check on the 58 explicitly
bilateral surfaces improves from **55/58 to 58/58**. The remaining 521 source
surfaces have no laterality verdict from this gate.

This is a placement correction, **not a topology or anatomy qualification**.
The cornea's compiled-coordinate quotient still has 39 exact intersecting
triangle pairs and is open/nonmanifold. The sclera still has 36 such pairs
and is open/nonmanifold. The lens-ligament source still has exact degenerate
triangles, so the pair predicate refuses it. These three remain among the
previous **22 per-surface geometry failures**. The precise post-excision
status, source identities, omitted face sets, SHA-256s, native command and
inspection packet identity are retained in
`Docs/media/right-eye-laterality-20260929`.

This gate covers gross side placement in one source-rest pose. It does not
prove eye component alignment, clinical anatomy, disjoint tissue regions,
physical volume/materials, cartilage, tendon or muscle mechanics, contact, or
whole-body function. BodyParts3D is CC-BY-SA 2.1 Japan and Z-Anatomy is
CC-BY-SA 4.0; the original pinned source attribution remains unchanged.
