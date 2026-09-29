# Compiled Numi Human surface census — 29 September 2026

The published ABI 5 Human source packet contains 579 surfaces and 1,289,680
triangles. We audited **every** original surface using its executed Float32
positions and indices. Two independent topological interpretations are
reported because the source commonly repeats vertex indices at identical
positions:

| Check | Closed and self-intersection-free candidates | Meaning |
| --- | ---: | --- |
| Exact native index graph | 26 / 579 | Strict packet connectivity; duplicate seam indices remain separate |
| Exact compiled-coordinate quotient | 546 / 579 | Identical Float32 positions welded for the audit, with no point moved and every source face retained |
| Selected native view with 11 previously proved repair copies | **557 / 579** | One visible representation per source member; all 598 raw and derived meshes remain in each native packet |

The jump from 26 to 546 is **a change in the audit's connectivity
interpretation, not 520 newly repaired meshes**. The 11-copy increase from
546 to 557 is the separate source-preserving topology repair. All 579 original
source meshes retain their original bytes, and the chosen 11 copies passed
exact source-face ancestry and exact Float32 self-intersection checks.

The authored-coordinate source quotient marked 551 surfaces closed. The
independent compiled-coordinate quotient made the **same closed/open decision
for all 579**. Exact triangle predicates then found five closed meshes that
self-intersect, leaving 546 closed embedded source candidates. Six meshes
have exactly degenerate triangles, so the triangle-pair predicate refuses
them rather than reporting a misleading zero. In all, the compiled quotient
completed the exact pair check on 573 surfaces. The native index graph and
compiled quotient are bound to the same source geometry hash for every
surface. Five real native inspection packets preserve all 598 source and
derived meshes, including the surfaces hidden for inspection.

The selected view still has 22 unqualified individual source representations:

| Group | Remaining | Examples and exact failure |
| --- | ---: | --- |
| Organ components | 8 | Hepatic segments III and IV, ventricular wall, three taenia coli members, corpus cavernosum and glans components |
| Lung lobes and pleura | 6 | All five lobes are open or nonmanifold; the closed pleura has seven exact self-intersecting triangle pairs |
| Eye and brain regions | 6 | Four ocular-region members and two neural-region members retain open, degenerate, or intersecting geometry |
| Vessel and duct | 2 | One liver portal-vein region and one epididymal duct reference self-intersect or remain open |

The exact stable IDs, source member IDs, layer codes, and failure categories are
in the selected-view report. These failures include the eight candidate
copies that were retained in the packet but hidden after self-intersection
audit; their raw parents remain visible. Some source families represent
aggregate and descendant anatomy simultaneously, so per-surface results
cannot be summed into disjoint tissue volume.

This is **per-surface geometry evidence only**. The census does not check
between-organ collisions or nesting, clinical placement, anatomical
boundaries, component containment, mass ownership, constitutive material,
physiology, contact, or whole-body mechanics. A watertight single surface is
not an admitted physical organ volume. The bilateral patellar centroid check
from the prior knee work also does not establish cartilage-facing orientation
or loaded articular contact. The 10-second standing video predates this
anatomy audit.

The independently rerunnable evidence is in
`Docs/media/whole-body-embeddedness-20260929`: two deterministic archives of
all 579 exact per-surface rows, a selected-view report, identity and command
receipts, and SHA-256s for every published file. The gate reads the archives
without extraction, validates every row and current predicate source hash,
rechecks each original compiled mesh slice against both censuses, and checks
all five retained native packet profiles. The source and candidate payloads
remain separately pinned. BodyParts3D is CC-BY-SA 2.1 Japan and Z-Anatomy is
CC-BY-SA 4.0; the pinned source attribution remains in the payload manifests.

The later [exact cross-surface abdominal organ gate](ABDOMINAL_ORGAN_SEPARATION_20260929.md)
tests a separate missing claim. Although five named organ representations
each pass this census's individual embeddedness gate, four of their ten
pairwise relationships contain a total of 182 exact triangle crossings.
That finding blocks disjoint organ-domain admission for the tested set;
the remaining anatomy surfaces still lack a comprehensive cross-surface audit.
