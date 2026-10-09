# Left vastus lateralis one-edge refinement screen

This is an offline conforming-refinement diagnostic for NHSKIN source row 64 (left vastus lateralis). It tests the longest edge of the most distorted mapped vertex-linear triangle at source face 5198 while preserving the existing local-frame and route-weight owner. The prior 4x4 continuous-field samples found positive local tangent-area ratios even where the mapped corner triangle was nearly degenerate and had an opposite normal; those samples are diagnostic, not a formal injectivity bound.

The marked source edge is local vertex IDs 2597–3054, length 27.1914 mm. It is shared by exactly source faces 5197 and 5198, with no other face using the same indexed or exact-coordinate edge. The edge midpoint is appended as one Float32 source vertex; the existing nearest-route-node inverse-square rule supplies its three weights. Both incident faces are split with their original winding. All old position, normal, binding-index, and weight records are byte-identical. Midpoint packing error from the exact binary64 midpoint is 15.8 nm. No source mesh or runtime asset was edited.

The full exact self-intersection scan was run on the coordinate quotient for the raw source and three saved poses. The unrefined baseline has zero source intersections and five unallowed self-pairs at each capture: (5195,5197), (5196,5197), (5197,5579), (5197,5911), and (5197,5912). With the one predicted midpoint inserted, the candidate has zero exact self-intersections at raw source and at captured steps 0, 10,000, and 20,000. Quotient topology remains closed and oriented with three components and Euler characteristic 6; boundary, nonmanifold, orientation, degenerate, duplicate-face, and unused-vertex counts are all zero.

The candidate does not move any old vertex. Existing source-vertex CPU map reconstruction differs from the native capture by at most 0.241 micrometres across the 3,974 vertices. The new midpoint is an owner-predicted counterfactual position, not a native captured point; its distance from the captured endpoint-linear midpoint is 1.665 mm at each tested state. The source triangle area changes by 4.91e-12 m² and signed volume by 1.71e-14 m³ from Float32 midpoint packing only. The full mesh scans apply the same pinned exact binary32-lattice triangle predicate and exact coordinate quotient as the prior pair diagnosis.

This is evidence for one bounded remeshing candidate only. It is not a changed native asset, an actual post-refinement capture, a full-body clearance result, or a continuous-time proof. The 20-second captures are from 1191; the 40-second capture at step 20,000 is from separate flat-reference run 1201.

## Reproduction

On the Mac mini with the retained Python 3.13 environment:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python3.13 refine_and_audit.py
/Users/n/numi-human-prep-venv-20261005/bin/python3.13 audit_unrefined_baseline.py
/Users/n/numi-human-prep-venv-20261005/bin/python3.13 verify_route_weights.py
```

The first command writes the compact candidate NPZ and full topology/intersection report; the second runs the same exact audit on the unrefined baseline; the third independently compares all existing vertex weights with the same route formula and verifies the midpoint field.

## Pinned results

- Candidate NPZ: `left-vastus-lateralis-one-edge-refinement.npz`, SHA-256 `106801ec7392cdb72e6a20407930f54a07ec81a84d19bb6c9661090a34afa9ff`.
- Candidate and source/pose audit: `report.json`, SHA-256 `7c844e6fae4e494907e2348e8bd91c9e26a56ea82f185a7589965bd3d9500d83`.
- Unrefined exact baseline comparison: `unrefined-baseline-report.json`, SHA-256 `4e50d1f5d5cf208f18d381b39557cb46cc73f87a225ce275735af0d408e0fdf2`.
- Route-weight validation: `route-weight-validation.json`, SHA-256 `a3ffe92cd79cd9acb6cd99a1f66fa2deee0db57bcdddcf8a7322b15ef0aed57f`.
- Exact predicate owner: `cardiac_cavity_intersections.py`, pinned in `report.json`.
- Execution dependency reconciliation: `execution-dependency-reconciliation.json`. The imported `common_atlas_skin_clearance.py` bytes matched committed Human blob `e0aa3fa` when these runs completed; that shared working path was modified afterward, and the later bytes were not used by the scans.
