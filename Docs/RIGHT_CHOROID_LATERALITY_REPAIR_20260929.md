# Numi Human right-choroid laterality correction — 29 September 2026

The complete 579-surface geometry census left placement unqualified. A new
source-rest laterality gate checked the **full selected triangle support** of
58 clearly bilateral kidney, eye, lacrimal, adrenal, testis, ureter and lung-
lobe surfaces. All 58 area-weighted centroids were on their named side, but
four right-eye source members had small contralateral pieces: FJ1337, FJ1340,
FJ1368 and FJ1371 (stable IDs 412, 413, 426 and 427). A centroid-only check
would have missed every one. The remaining 521 surfaces were not given a
laterality verdict; left/right labels in the heart and liver do not imply an
entire surface on one side of the body midline.

The selected right choroid, ID 587 derived from original source ID 412,
contained a separate **eight-triangle, six-vertex** component across the
source-rest sagittal midline. Its exact source OBJ ancestor faces are
28862–28869; these eight faces are 0.0000991% of the selected mesh's surface
area. Excluding that one entire edge-disconnected component yields a single
closed, exact self-intersection-free 28,852-triangle component. No retained
vertex moved and no face was added. The 579 original surfaces and all 19
earlier candidate copies remain byte-identical in a new 599-surface ABI 5
payload; the corrected inspection copy is ID 599. This is **a source-face
excision**, not the source-support-preserving operation used for the earlier
11 copies. The manifest names every excluded face and every retained parent
vertex and triangle.

One actual native source-rest packet retains all 599 surfaces and hides ID
587 while displaying ID 599. Its source-rest body poses are identical to the
earlier packet. The independent native packet audit checks owner, stable ID,
local vertex and index bytes, instance ranges and visibility for all 599.
The selected bilateral full-support count rises from **54/58 to 55/58**;
FJ1340, FJ1368 and FJ1371 still fail. ID 412 already passed the previous
per-surface geometry gate, so the prior 557/579 geometry-only count and its
22 geometry failures do not change. This correction closes one newly found
spatial defect; it does not qualify the other eye meshes.

Reproduction is bound to the published parent payload SHA-256
`c8b8a01aac9d55d1dd154f0e3f92450e5a1a980668eebfcc2a012bb47268b280`,
the verified 579-source manifest chain, both complete census archives, the
published selected-surface gate and source-rest pose, and the retained native
packet. The new payload SHA-256 is
`c32c13c555308f9671c92f1083db4f9fa4a4b096cd3d1734412013c0c7a47ecd`.
Command, source hashes, manifests, the full 58-row laterality report, native
audit, and source-rest images are retained in
`Docs/media/right-choroid-laterality-20260929` and
`Build/right-choroid-laterality-repair-20260929`.

This check is for gross source-rest laterality of 58 named surfaces. It does
not establish internal eye component placement, clinical anatomy, cross-
surface disjointness or containment, tissue volume/material, contact,
muscle/tendon coupling, or physical mechanics. BodyParts3D is CC-BY-SA 2.1
Japan; Z-Anatomy is CC-BY-SA 4.0. Their pinned attribution is unchanged.
