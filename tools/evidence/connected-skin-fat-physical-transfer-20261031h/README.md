# Skin/Fat quality-repair increment 678

This package preserves an **historical, unadmitted** quality-repair snapshot and its post-622 input lineage. It is not the live checkpoint. Quality work continued after this snapshot and remains in its own current checkpoint; the historical 678 files here do not represent that later state.

## Frozen 678 result

The paired boundary-diagonal batch and metadata-only finalizer produce:

- Skin: 725,266 tetrahedra, 448 native-quality failures, 0 nonpositive cells, exact 389,192-face target boundary.
- Fat: 772,668 tetrahedra, 2,531 native-quality failures, 0 nonpositive cells, exact 445,976-face target boundary.
- Exact changed-surface context: PASS against current O/I/D, both walls, and all 185 protected surfaces; 0 crossing pairs.
- Independent paired-volume replay: PASS.
- Physical admission: false. Accepted native steps: 0. Regional mass allocation: pending.

The metadata finalizer changes no geometry. It rebuilds source-boundary row metadata to match the current typed O/I/D targets. The finalizer report and independent readback are included. The final state retains an inherited `PENDING_METADATA_REBIND_NO_GEOMETRY_CHANGE` status string; the versioned finalizer receipt/readback separately records the successful row-aligned target reconstruction and exact actual-boundary match.

## Source and history

The accepted 678 snapshot pins `boundary_edge_flip_676_local.py` v1 at SHA-256 `4f1fc00a95b3caaca947fcc8fd3c13cb76b4ef6d968892d5bafdfced6e85739d`. The separately preserved v2 copy is not substituted for v1. `boundary_edge_flip_676_import_window_receipt.json` records the earlier 681 import-window history and limits what can be attributed to each copy.

The package includes the 679/680 drivers, their local owner/metadata finalizer, controls and fixture receipt, exact 678 context/source audit, independent replay, and finalizer readback. `failure-receipt.json` preserves rejected-operation counts, the partial 677 finalizer attempt, nonzero quality gates, and the source-window caveats.

## Restoring the data

The content-addressed archive is `quality678-post622-lineage.tar.zst`, attached to GitHub Release `human-connected-quality-678-20261031`. `archive-index.json` lists every source path, member, SHA-256, and size. It deduplicates repeated files by content hash. The archive contains the 678 snapshot, both finalized volume NPZs, typed O/I/D sources, and the post-622 parent-state chain. Checkpoint 622 is an explicit dependency on the existing `connected-skin-fat-physical-transfer-20261031g` package; its geometry is not duplicated here.

After downloading, verify the release-asset SHA in `manifest.json`, then inspect and restore the archive using `restore_archive.py`. No source files were deleted or cleaned up during publication.

This evidence does not establish zero native-quality failures, source-mass conservation, physical admission, or a clinical claim.
