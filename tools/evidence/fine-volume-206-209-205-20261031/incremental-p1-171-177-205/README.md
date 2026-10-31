# Incremental source-only P1 and node-map reuse

This private tooling preserves 171/177 work when a future candidate changes only a subset of tetrahedra and keeps the point coordinates fixed.

- `incremental_p1_reuse_205.py` reuses a 171 row only for an exact ordered `(material, corner0, corner1, corner2, corner3)` match after a one-to-one exact-F32 candidate-node-to-base-node map. Any corner permutation, including an even cyclic permutation, is classified as changed so corner-indexed fields are regenerated instead of silently copied.
- Every changed candidate row must have a complete output bundle from `export_changed_p1_rows_205.py`. That producer recomputes the integrated max-share group and all P1 parent/semantic unions from the four authenticated node-group support records, then makes the P1-supported nodal reference-anchor selection. It does not transfer the retired cell's scalar label or call a group tuple geometric ancestry.
- A regenerated row has `p1_support_cell_row = -1`: it has no prior cell-row identity. Its parent rows and semantic tuples are computed anew from node support. The `.source.csv` fields remain a separate centroid projection diagnostic; the label/projection CSR carries the nodal P1 compatibility anchor.
- If every candidate node maps one-to-one to the base 177 node rows with byte-identical F32 coordinates, the merger reindexes every 177 record and preserves its fields. It validates each record's `source_partition` and role-local node ID against the authenticated combined-node-map crosswalk, then checks the corresponding skin or fat row in the material-specific 171 node-group CSR. It does not reinterpret 177's geometric-donor/region scalars as complete P1 authorization tuples.

The scripts and receipts are source metadata utilities only. They do not change or certify geometry, allocate mass, run native steps, admit a physical mesh, or qualify motion. Full changed-candidate stitching and 177 reindex for a changed candidate were not run because no new measured candidate was supplied in this pass.

## Bounded validation

`actual-parity-001/` regenerates 16 rows sampled across Skin (material 15) and Fat (material 16) from the unchanged 142 candidate. All 24 per-cell P1 fields except the intentional `p1_support_cell_row = -1` sentinel, all 208 CSR payload slices, and all 57 CSV field values per row match the authenticated full 171 output exactly. This is a reproducibility control for unchanged cells, not a changed-geometry qualification.

`baseline-role-material-support-correction-001.json` replays the role-specific 171 node-group checks over all 323,602 nodes / 448,970 177 role rows. It supersedes `base-177-incidence-recheck.json` as a diagnostic: that earlier check incorrectly broadcast one 177 donor/region scalar to every incident material bit, producing 30,294 false missing-tuple reports at 15,147 interface nodes. The corrected check preserves those scalars and validates each skin/fat source row against its own material-specific 171 node-group CSR. It is source-only and does not assert mass or physical admission.

`test_incremental_p1_reuse_205.py` contains ten tests for ordered identity, permutation handling, CSR stitching, exact F32 node reindex, role-preserving interface records, material-specific P1 support, wrong-role rejection, and cross-material CSR rejection.
