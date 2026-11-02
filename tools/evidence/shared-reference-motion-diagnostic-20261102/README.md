# Shared-reference compensated-motion arithmetic diagnostic

This package preserves a CPU-only diagnostic of the shared compensated-deformation arithmetic used by the Numi Human engine. It compiles the unchanged C++ helper from the owning engine checkout, replays a compact F32 rest-geometry fixture, and checks three rigid-frame cases.

## Result

The retained fixture contains 4,329 tetrahedra: the union of rows flagged by three earlier materialized-F32 probes. It is a failure-focused subset, not an unbiased sample of the whole volume.

- Z translation by 1 mm: J range 1.0 to 1.0; 0 invalid values.
- Z rotation by -1 degree: J range 0.9999403437617088 to 1.0000495818370008; 0 invalid values.
- Z rotation by +1 degree: J range 0.9999503401369085 to 1.0000599702680029; 0 invalid values.

Independent source review checked the row-major inverse-rest/cofactor and edge construction against the compiler's matrix layout. It found zero mismatches in the post-F32 inverse coefficients across the retained 4,329 cells. The reviewed Metal path uses safe/precise math modes and generated Metal commands use no-fast-math. Those source checks do not establish GPU arithmetic parity.

## Replaying the compact fixture

Run tests/test_fixture_replay.py with Python's unittest. The test compiles source/shared_reference_motion.cpp with the exact pinned external headers and flags in header-closure.json, then replays fixture/selected-rest-f32.bin. It does not load the original 75 MB element file or the source node file.

producer/run_diagnostic.py is the original generator used for this evidence. Regenerating the selected rows requires the original 1020 probe CSVs and source mesh paths listed in package-pins.json. Those large source-mesh files are referenced by path and SHA only; they are not copied here.

## Dependencies, ownership, and license

The engine headers are not copied into this evidence package. Their exact paths, hashes, include chain, and owning repository revision are pinned in header-closure.json. The owning repository has a pinned THIRD_PARTY_NOTICES.md; no root LICENSE file was present at that pinned revision. This package does not create or imply a separate license grant.

## Limits

This is a CPU instantiation of shared C++/Metal arithmetic, not a native or GPU run. It advances no native simulation steps and makes no physical-admission claim. The rigid displacements are synthetic, split into F32 high/low words; there are no forces, contact response, or controller updates. The actual GPU state still requires native verification.

The pinned producer is copied byte-for-byte from the executed source. Its upstream file has an extra blank line at EOF; that whitespace is retained for hash fidelity.
