# TetGen hull-to-real region attribute correction

This source-only increment addresses a TetGen v1.6.0 cell-attribute propagation defect: hull flip routines reused slots whose attributes were initialized for hull cells, so a hull-to-real transition could leave region attribute zero on otherwise in-domain cells. The fix snapshots the complete attribute tuple (and variable-volume bound) from the real tetrahedron opposite the old hull face, restores it only when a hull slot becomes a real cell, and marks missing-owner or unprotected real-interface disagreement as invalid rather than guessing. It does not change coordinates, predicates, or flip decisions.

## Rebuild and focused controls

To reconstruct the derivative from the pinned Human 054 source, copy that source to this package’s `tetgen-hull-attribute-fix-minimal.cxx` path and apply `patch -p1 < tetgen-hull-attribute-fix-minimal.patch` from the repository root. The patch updates only the copied package file. For a rebuild, set `TETGEN_VENDOR_DIR` to the pinned TetGen v1.6.0 header/dependency directory and run `build_controls_variant.sh`. The script checks the Human 054 base-source, header, and predicates-object SHA-256 values before compiling. It runs six focused controls and compares their output with the checked-in golden stdout before compiling the fixed TetGen source.

The controls cover complete tuple and variable-volume inheritance, missing/malformed owners, unchanged non-transition slots, fail-closed mixed labels at unprotected real interfaces, preservation of distinct labels on protected interfaces, and accepted matching labels. The clean derivative and its compact diff are included alongside the helper and tests. The existing evidence-level `.gitattributes` has byte-preserving rules scoped only to this derivative and its whitespace-bearing diff.

## Frozen replay evidence

`build-and-control-receipt.json` records source/dependency hashes, test count/output, and replay results from the fixed executable. On both unseeded inputs, generated node coordinates, tetra connectivity, and labels exactly match their separately instrumented reference output. This isolates the attribute repair from the geometry decisions for those runs.

The original Fat replay contains 31 cells labeled with automatic region IDs 17–22 even though its 84 declared regions all carry attribute 16. `auto-region-to-material-certificate.json` records a scoped semantic interpretation of those labels: TetGen assigns incrementing attributes to uninfected connected components, all 31 component-cell centroids lie inside the matching closed Fat shell, and each component has a verified seed carrying the declared attribute 16. Thus 17–22 map to material 16 for that exact input; they are not independent material IDs.

The six-seed Fat replay is a separate mesh. It changes connectivity and Steiner coordinates, and the receipt preserves that difference rather than claiming a cell-row correspondence. Its output has 16 zero and 89 negative Float32 oriented cells and 210 extra boundary faces. The unseeded geometry-identical replay still has 8 negative Skin cells, and 16 zero plus 93 negative Fat cells. Neither is a valid physical volume.

## Limits

This evidence validates one focused code fix and its behavior on two frozen input replays only. It does not establish acceptable mesh quality, a valid paired Skin/Fat volume, source330 mass transport, native import, physical installation, or runtime admission. `physical_admission` remains false and `runtime` remains zero in every receipt in this package.
