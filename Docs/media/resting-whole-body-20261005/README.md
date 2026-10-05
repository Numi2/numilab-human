# Resting whole-body anatomy evidence — 2026-10-05

This directory keeps internal verification evidence for a registered thorax
build and a native supine Human run. It is not a board deliverable.

## Right inferior lung patch

The current source-bound build compiled 397 anatomy surfaces into ABI 5, with
1,047,118 vertices and 4,893,798 indices. Its payload SHA-256 is
`e6894609f9867f611db612d3e9474533fbc949be7a3a6c3a9b75cd9ec0c84a8d`; the
receipt and manifest are retained under `anatomy-compile/`. The exact local
source hashes used by the Mac mini build were `a188dabe6e9377b86467351e5724fad8bd375420a96a8129a6ca869dffe4283e` for
`resting_anatomy.py` and
`837f6281d6f2c58e4aa2e8d1972af7cb67ce068bc25862504eebc0f82df48a38` for
`surface_topology_audit.py`.

For stable surface 306, “Inferior lobe of right lung,” the native Float32
registered mesh had eight exact self-intersecting triangle pairs. The build
replaces 18 source faces with a 14-triangle derived cap using existing boundary
vertices. It moves no retained coordinates; the retained-coordinate hashes
match. The result is closed and oriented with Euler characteristic 2, and the
same exact Float32 predicate reports zero remaining self-intersection pairs
for this surface. Surface area changes by `-8.03e-6` relative and the signed
envelope volume by `+1.93e-7` relative. This closes only the named surface's
self-intersection check. The whole-payload receipt still leaves other
self-intersections, lobe interfaces, pleura interfaces and organ interfaces
unassessed.

The two pairwise lobe audits use the same compiled payload hash but disagree.
The older `lobe-pair-audit.json` integerized each lobe on its own scale before
calling the cross-surface predicate, so its coordinates were not in one shared
spatial frame. Preserve its 306:309 count of 306 as a superseded raw outcome,
not as a valid cross-lobe measurement. `lobe-pair-audit-common-scale.json` uses
the shared denominator `4398046511104` and reports 2,299 intersecting triangle
pairs across 305:308 (989), 306:307 (690), 306:309 (177), and 307:309 (443).
Those intersections remain open anatomy gaps; no lobe geometry was changed by
the audit.

## Native integrated run

The first attempt stopped at 64 ms with the generic anatomical-volume ownership
failure. Its final recorded surface sample had non-finite right-atrial shape
coordinate `q_ra` and 11,556 non-finite skin vertices. The attempt has no
invocation manifest, so retain it only as failure evidence.

The second attempt used its own hash-bound invocation and accepted 6.000000285
simulated seconds over 6,000 one-millisecond steps on an Apple M4 Pro. It
recorded seven complete filling/ejection cycles, 497.33 mL of aortic ejection,
490.47 mL of pulmonary ejection, one reported breath, and a maximum tidal
volume of 552.82 mL. Across 189 anatomy
audit samples, all cardiac shape coordinates were finite, no skin vertices
crossed the 1 mm inspection limit, and maximum recorded chamber-volume error
was `1.17e-6`. Blood-volume error peaked at `0.002794 mL`; root assistance was
zero. The terminal root speed was `0.00137 m/s` and support force was `675.03
N`. The observed maximum generalized acceleration was `582.44 m/s^2`, and
maximum penetration was `0.120 mm`. The six-second run is bed-supported and
supine, not the requested ten-second standing run. Its log ends with
`presentation_qualification=pending`.

This is software-level coupled circulation, respiratory and anatomical
presentation evidence, not biological or patient qualification. The chamber
surface coordinate is derived from the hydraulic volume; this run does not
demonstrate cardiac electrical conduction or an electromechanical myocardial
force owner. Most importantly, the invocation binds the consumed candidate-
final anatomy payload to SHA-256
`3f7659c6d99c8aedb04c09543477ff5377a6f7471a0d89ff24f0ce4bde7bb2e9`, not the
new lung-patch payload above. Its separately hashed anatomy-complete producer
input is `0b57ec1c79cea055a72de4c9555fca36ce04f3bf1a344dd15d32e577d6056d83`.
Neither hash matches the repaired payload, so do not use this run to claim
native integration of the patch.

The native invocation, run log, coupled trace and surface audit CSVs are
retained in `native-002/`; the earlier failure traces are in `native-001/`.
The full NHA payload, MRVPack and movie remain on the Mac mini under
`/Users/n/numi-human-resting-evidence-20261005/`; no board visual was created.
