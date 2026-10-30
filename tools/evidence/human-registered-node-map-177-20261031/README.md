# Current typed 217 registration, run 177

This bundle preserves the source and compact receipts for the full child-cell label export and combined-node registration. It is source-preparation evidence only; it does not establish physical admission, mass mapping, or native runtime behavior.

## Run 177

- Status: `REGISTERED_NODE_MAP_PREPARED_SOURCE_ONLY_NOT_PHYSICAL_ADMISSION`.
- Node map: 448,970 rows for 323,602 volume nodes; roles: `{"D": 120636, "I": 125368, "O": 76669, "interior": 929}`; shared I interface: 125,368 nodes.
- Historical parent support: 0.02 m; ambiguity tolerance: 1e-08 m.
- Capacity profile: `full-reference-500k-nodes-2m-cells` (500,000 nodes / 2,000,000 cells).
- Physical admission: false; mass mapping: `NOT_RUN`; runtime: 0; accepted steps: 0.

## Source and controls

- Full label exporter: `sources/source-007/export_child_semantic_labels_217.py`, SHA256 `e2075c47e867dc56629d066db8b4409c557b0b8bf1069ba192aa2108355c7910`; its exact launch is preserved beside it.
- The source-008 compatibility adapter verifies bundle 143 and the 713 inherited-plus-delta geometry scope, then delegates to the source-005 217 owner. Source-005 dependencies are included and hash-pinned.
- Source-008 unit controls: 10/10 pass. See `receipts/source008-tests.json` and the captured test output.
- CLI limits used: `--maximum-parent-distance-m 0.02`, `--ambiguity-tolerance-m 1e-8`, and `--capacity-profile full-reference-500k-nodes-2m-cells`. Adapter inputs include node map/node/element/labels/parent base, bundle receipt, geometry scope, registered source directory, poses, and output directory.
- This CLI prepares registration only. It does not run mass transport or a native simulation.

## Native handoff files kept local

- Registration map: `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv`; SHA256 `40646a6c501b89046e0e9836790a78b3303ad8f0033fe5281fc64ebb760fccc3`; 223,309,676 bytes.
- P1 node-anchor projection CSV: `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/p1-anchor-projection.csv`; SHA256 `b0b09a9bbd46b27a6763b82fa96c87c16e5c9a0649bcece769bfd3f4a5737870`; 564,998,749 bytes; 1,596,903 child rows.
- P1 tie and semantic-union CSR: `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/p1-anchor.npz`; SHA256 `4fe982548d0cfc4662e29f244b9f616daa21b39c05b5e4a5ef81f0ac6a80d985`; 286,586,668 bytes.
- Five-column labels: `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/child.labels.csv`; SHA256 `27b7620e328578b6521f341cdda9962412134bf6e8a826e7bc805c0063617685`.
- Centroid `.source.csv` witness: `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/combined-child-handoff-142/semantic-labels-171-vectorized-full-export/child.labels.csv.source.csv`; SHA256 `222e9964d186e57a346bdf06a4226e1ea680be39b24f35e499177070374ce5de`. These centroid witness fields remain distinct from the nodal P1 anchor.
- The large registration map, P1 projection/CSR, child tables, and combined node map are intentionally not duplicated in Git. Their full local paths, sizes, and hashes are in `manifest.json` and the registration receipt.

- Geometry scope receipt: `receipts/geometry-scope-713.json`, SHA256 `35d6a46efac809166d2ef0ea6d5860391c78f2e2ecff2466ac8b2b794ed1ed34`, status `PASS_GEOMETRY_SCOPE_INHERITED_PLUS_EXACT_DELTAS`. It retains the coverage chain and does not rewrite incomplete historical flags.
