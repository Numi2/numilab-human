# Complete Skin geometry evidence (073)

This bundle preserves the Skin volume-construction sequence and the typed paired-shell preparation that feeds the existing 217 owner. It is a geometry and source-ancestry increment only.

## Evidence

- The 068, 069, and 070 reports preserve partial attempts, their exact proposals, and available component records. They are not promoted: their receipts record remaining negative cells, missing PLC facets, or boundary mismatches.
- The 072 batch proposal has 202,525 nodes and 743,497 material-15 cells. Its report records zero F32 zero/negative cells, zero missing original PLC facets, zero extra boundary facets, and zero nonmanifold tet faces.
- The 073 source004 owner readback writes `skin.node` and `skin.ele`; it verifies positive F32 cells, exact original PLC points, exact oriented PLC boundary, and candidate/source volume difference of 3.2395960913866872e-16 m^3.
- The 074 existing-owner preparation emits exact typed Skin and Fat shells and full per-face ancestry. Its receipt records 202,037/404,114 Skin vertices/faces and 245,996/492,956 Fat vertices/faces, closed manifold shell edges, exact parity to the 010 PLCs, and exact shared-I geometry. The preparation completed in 19.276 seconds.
- The parent 038 bundle at commit `1326a513588d0decd05aa774cbd30845ed10888c` carries the upstream source operation history. `parent-links.json` links the current full-surface/all185 geometry checks to that parent evidence.

## Qualification boundary

These results establish geometric construction, owner readback, and typed shell ancestry. They do **not** establish conservative mass transfer, native installation, runtime motion, accepted steps, or physical admission. The 073 receipt records `mass_mapping: NOT_RUN`, `native_install_and_motion: NOT_RUN`, `physical_admission: false`, and `accepted_steps: 0`. The 074 receipt is geometry-only and also records `physical_admission: false`.

The 073 volume is Skin material 15 only. The 074 package supplies paired Skin/Fat boundary shells, not a paired Skin/Fat volume mesh. Density, tissue mass, donor transfer, native scene, and runtime claims remain open. Intermediate partial or failed evidence is retained as such.

## Reproduction and rights

The frozen owner scripts are under `reproducibility/`. They reference task-owned source roots on Numi and use the existing 217 owner. The receipts pin source paths and hashes; the file manifest also records each source file SHA-256, size, mode, UID/GID, and mtime. No new license is asserted. Existing ownership, privacy, and use restrictions continue to apply.
