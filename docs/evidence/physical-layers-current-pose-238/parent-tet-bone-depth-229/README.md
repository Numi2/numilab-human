# Parent tet / bone depth diagnostic

This compact packet retains the current-pose two-region exact diagnostic and its source. The full 185-bank audit was not run.

- Native compiled cell mesh: 7,233 nodes, 28,632 tetrahedra (14,316 each in materials 15 and 16).
- Selected typed-bank regions: (59, 95), worst stored I witness; (92, 15), worst stored D witness.
- Exact parent tetra-face intersections: 615 and 536 pairs. The material 15/16 interface has 41 pairs in each region.
- The selected bank triangle soups are closed and manifold; both sampled layer depths remain negative by millimeters after replacing pinned 039 body positions with the native run all_bodies poses. This is local evidence of actual containment, not a clinical or whole-scene qualification.
- See report.json for exact tet row, donor, anatomical region, stable source face, parent node, and bone triangle identities. manifest.json records source, input, bank, pose, exact-owner, command and report hashes.

The earlier attempt-006 report is retained in its original diagnostic directory and used only to show the initial stale-pose comparison. The current-pose report in this folder supports the conclusion above.
