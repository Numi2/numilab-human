# Source-linked classification of the 925 lung-seam events

This evidence-scoped check asks whether the 93 exact native intersections retained by the 927 audit are tied by exact source topology to their declared reciprocal fissure patches, and whether their normal excursions cross the source-established side by more than the existing pair half-ULP coordinate bound. It does not change the geometry, suppress an intersection, or change an audit gate.

The result is 93/93 topologically linked to the declared patch set: 47 event faces share an exact reciprocal-patch boundary edge, 40 share an exact boundary vertex, and 6 are themselves faces in the 307/309 reciprocal patch. Those six are inside the reciprocal patch, not on its outer boundary. The link is established from exact float32 coordinate keys and triangle face incidence; there is no distance threshold. Each captured intersection point also maps inside the captured triangles and back to the corresponding NHA source face. Source-side classification is 90 positive, 3 negative, and 0 ambiguous. The largest native excursion to the opposite source side is 10.789547644 nm, or 0.45621242 of that event's existing pair half-ULP bound.

This supports a scoped description of source-patch-linked, quantization-bounded apposition for these 93 events. The exact native intersections remain present. The classification does not establish anatomical intention, absence of contact or penetration, global lung clearance, or that coordinate rounding alone caused the native intersections. The half-ULP estimate does not bound respiratory deformation, GPU skinning arithmetic, or registration uncertainty. This result is not a full all-pairs anatomy qualification.

## Pinned inputs and method

`report.json` pins the composed NHA 924, the 927 targeted native events and reciprocal patch map, the 927 source/native face-pair comparison, the 931 signed-range report, owner parsing and reciprocal-map code, and the eight 925 MRVPACK/receipt pairs. All hashes were rechecked during `analyze.py`. The parser is the existing `resting_anatomy_interface_patch.parse_payload`; exact reciprocal face maps follow the float32 coordinate-byte and opposite-winding implementation in `resting_lung_edge_repair.py` (`shared_face_maps`, around line 1216). No owner or runtime source was edited.

For each pair event, the script verifies NHA/source face vertex bytes, reconstructs exact reciprocal faces and their boundary edges, and maps every native intersection point through captured-triangle barycentric coordinates. An event links to the declared patch only if an event face is itself a reciprocal patch face or has an exact source-coordinate edge/vertex in that patch boundary. Closest-footprint distances to the boundary are recorded descriptively and never decide acceptance. The source-side deadband is 1e-14 m; classifications are unchanged for deadbands from 0 through 1e-12 m. This is a numerical classification deadband, not a biological clearance threshold. Ambiguous side, unallowed source overlap, missing patch topology, a new face-pair witness, incomplete scan coverage, self-intersection, or an excursion above the pair bound fails closed in the reusable validators/tests.

## Reproduce this 925 check

From macmini, run:

```sh
cd /Users/n/numi-human-resting-evidence-20261005/native-lung-seam-patch-linkage-933
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/n/numi-human-prep-venv-20261005/bin/python analyze.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v test_patch_linkage.py
```

The analyzer verifies the full SHA-256 of all eight existing 925 packs and matching receipts before reading the pinned 927/931 event records. The tests cover the exact 93-event patch-link census, all source-side and bound outcomes, and fail-closed synthetic cases for ambiguous signs, out-of-bound excursions, disallowed source overlap, non-patch events, incomplete capture coverage, self-intersection witnesses, and unknown face-pair witnesses.

## Rechecking later captures without inheriting the 927 neighborhood limit

The original `native-lung-seam-structural-audit-927/audit-922-native925-detailed.py` only scanned 21 retained witness neighborhoods plus collapse-operation changed stars. Its successful exit or `native_targeted_coverage_complete` flag is not full-row coverage. For a new accepted capture (including steps 10000 or 155000), make a fresh evidence copy of the existing 927 driver, pin the exact new invocation/pack/receipt hashes and unchanged NHA SHA `3c444be7736c066a992988cc32b687917e1c4c5c3968a16b4d5f0106d5b5024e`, and use its existing `native_rows`, `make_records`, `scan_spec`, current exact predicate, and accepted-pack reader. Do not edit the retained 927 driver or treat its old witness list as exhaustive.

For each new capture:

1. Decode and receipt-check every mapped row 305–311: semantics 51023 for 305–309, 51024 for 310, and 51010 for 311. Require source face order and source vertex mapping to match NHA 924.
2. Run the exact same-surface predicate across every face in each row 305–311. Build records from every face ID and call the existing exact `cardiac_cavity_intersections._audit_pair(records, records, same_surface=True)` helper (or the same `triangle_intersection_points`/`_allowed_shared_point` path used in 927). Require a complete result with zero unallowed self pairs; preserve any failure pair IDs.
3. Run the existing 927 `scan_spec` across every face of each of the eight declared reciprocal interfaces: `(305,308)`, `(305,311)`, `(306,307)`, `(306,309)`, `(306,311)`, `(307,309)`, `(307,311)`, `(308,311)`. For each pair pass `candidate_face_ids` equal to the full `range(face_count)` for both owners, not the retained 21 witness stars. Preserve the entire `unallowed_intersection_witnesses` array and exact allowed reciprocal-face/shared-vertex-edge counts.
4. Add a per-pose `full_scan_coverage` receipt with the exact source NHA SHA, `all_faces_considered: true`, self rows `[305,306,307,308,309,310,311]` and their source face counts, all eight cross pairs and their two face counts, accepted step, MRVPACK SHA, and matching receipt SHA. Call `validate_full_scan_coverage` from this package; it rejects omitted rows, omitted or duplicate pairs, changed source topology, and incomplete identities.
5. Generate the matching signed-range output with a fresh evidence copy of the existing 931 `analyze.py`, changing only its evidence input/output paths and capture-step list. Keep its source orientation and projected-range computation unchanged. Pass all raw unallowed cross witnesses to `reject_unreviewed_witnesses` with the pinned 33-key source face-pair catalog in `report.json`. A newly appearing owner/face pair is rejected for explicit source/topology review; it is not silently accepted as “roundoff.” Call `reject_native_self_witnesses` on all full-row self results. Then apply the same source-side and half-ULP checks to every event. Do not filter a witness because it is small.

This is an execution recipe only. No 10000- or 155000-step full-row scan is included in this evidence, and no claim is made about those later captures.
