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
self-intersection check. The common-scale lobe audit below separately finds
2,299 cross-lobe triangle intersections; lobe and pleura interfaces remain
open anatomy gaps.

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

## Patched-anatomy native integration and timestep comparison

The later `cardiac-geometry-binding-002/candidate-final` composition consumes
the exact patched anatomy payload above (`e6894609…c0c84a8d`) and emits the
native runtime payload `3f7659c6…de7bb2e9`. Its retained receipt
(`68a65c61…4419dda62`) records that the non-cardiac source records, including
the right-inferior-lung patch, are copied in source order; it also records
`mechanical_mass_or_volume_changed=false` and
`physical_solver_changed=false`. The two native runs below bind this composed
payload by hash, establishing software integration of the lung patch.

Both runs accepted 6.000000285 simulated seconds, seven complete hydraulic
filling/ejection cycles, one breath, and zero root assistance. The 1 ms run in
`native-004/` recorded 497.327652 mL aortic and 490.468403 mL pulmonary
ejection, a 552.824 mL maximum tidal volume, 581.67 m/s² maximum generalized
acceleration, and 2.40 µm maximum penetration. It took 121.52 seconds of wall
time. The 2 ms run in `native-dt2-002/` recorded 497.111992 mL aortic and
490.232778 mL pulmonary ejection, a 552.722 mL maximum tidal volume, 3,011.9
m/s² maximum generalized acceleration, 3.70 µm maximum penetration, and
706.03 N support force. It took 62.99 seconds of wall time, about 1.93 times
faster. The larger acceleration peak at 2 ms remains a meaningful cadence
limit despite close circulation and breathing totals.

The 1 ms binary SHA-256 is
`5f308c35c2c43ad40c54a33456d65677c6a40321dad0b4acbc174179648f9f62`; the 2 ms
binary is `f69d9d8215db84996e3f11e517893ddd84c32a06570b579b449a0fa0a4edf2a8`.
Both invocations bind runtime source revision
`f425e09a6d3a2971f107e31b75740c251c930103`. Their exact manifests, logs,
coupled traces and surface-audit CSVs are retained locally. The full composed
anatomy payload and rendered movies remain on the Mac mini.

An additional attempt to pass the intermediate `e689…` payload directly to
the native runtime stopped at admission with “lacks the source-bound cardiac
cavity ownership receipt.” Its metadata confirms no physical step ran. This
is a composition-contract failure, not a failed simulation; the source-bound
composition above supplies that receipt.

These are six-second, bed-supported supine software diagnostics, not the
requested ten-second standing result. The 2 ms acceleration peak and pending
presentation qualification remain open, as do the 2,299 lobe intersections,
pleura interfaces, cardiac electrical-conduction and myocardial-force
qualification, biological validation, and measured-subject qualification.
