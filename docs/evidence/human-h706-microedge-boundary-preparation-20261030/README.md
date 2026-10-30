# H706 microedge collar source preparation

This source-only increment adds `prepare_microedge_boundary_706.py` and its focused contract tests. The helper consumes the pinned H706 plan-003 manifest and bounded O/I/D microedge-contraction inputs, checks the parent plan inputs by SHA-256 (including the wall parent-cell and parent-node CSVs), reindexes the existing collar, preserves the ordered paired-rim triangles, and verifies the 149 protected-point routes and existing wall-provenance owner. Its output receipt records the helper SHA-256 as `owner_sha256`.

The fixed parent plan is `docs/evidence/human-h706-source-volume-20261030/plan/plan-003-manifest.json` (SHA-256 `1e4d83b2f3f6ac62d4c552881a967fdb205c7d001f701f26fd20447f64752bd4`). The helper keeps the inherited input hashes as gates; it does not rewrite historical protected-row classifications or claim that the changed surface coordinates are measurements.

## Focused tests

Run `python3 -B tools/evidence/human-h706-microedge-boundary-706/test_microedge_boundary_706.py`. The recorded test run used the Numi preparation Python environment and passed all 8 tests. The compact result, exact helper/test hashes and captured output are in `test-receipt.json` and `test-output.txt`.

## Scope

The tests exercise source-packet validation, contraction ancestry, preserved endpoints, collar reindexing, and tamper rejection for both wall parent CSV pins. This publication and its test run generated no geometry or volume mesh, performed no mass transport, and ran no native import or motion. It does not establish intersection-free surfaces, positive tetrahedra, mass/density admission, or physical Human behavior. No failed geometry candidate is included or promoted.
