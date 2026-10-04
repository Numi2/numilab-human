# Scan 002 — direct source-mask tetrahedralization

All 12 pinned scan-002 labels produced one tetrahedral volume candidate each: 3,984,429 source voxels and 23,906,574 tetrahedra. The original NIfTI was streamed from the registered TCIA archive; its decompressed SHA-256 matched the intake receipt, every label count matched, and all direct-mask boundary triangle multisets matched the registered surfaces.

The [compiler receipt](meshes/receipt.json) and [independent verification](independent-verification.json) bind all outputs. The independent verifier rereads scan 002 and every VTK file, verifies positive tetrahedra and valid face incidence, reproduces exact source-mask boundaries, and reports relative volume error from `1.13e-16` to `2.60e-15`. The verification JSON SHA-256 is `b65bb4720b29be6ce2cf637d639689a4e8e6669b01d2fddb129d981e6cfa3ee6`.

Scan 002 is an automatic segmentation source unit only. The receipt does not establish cohort identity, expert segmentation quality, Numi Human subject binding, physical organ/mass ownership, tissue material, mechanics, vessel lumen/perfusion, or physiology. The verifier checks tetrahedron face incidence, not full vertex-link manifoldness or mechanical suitability.
