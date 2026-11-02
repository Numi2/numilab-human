# Native horizon v16 and 1060 pinned input preflight

This source/evidence increment packages the direct-O first-run gate (010c), the bounded increasing-horizon controller (010b), the v16 motion verifier, and the source-only 1060 input preflight. Frozen owner files are copied without source edits. The v15 route test is copied with only its verifier filename and imported module name changed to exercise v16; the publication receipt verifies that derivation.

## What is covered

- 010c checks the fresh orientation audit and owner SHA pins before creating a job directory.
- 010b enforces the declared 2, 16, 128, 1024 increasing-horizon progression from the same pinned baseline, records the bounded capture roster, and validates terminal/performance markers.
- v16 verifies the direct-O observer route and requires a fresh signed surface-orientation audit binding. The included owner is 1057, SHA `9c9a68a7d1122e326b7504737a8d3abeba9d3c1a1ddb011f6320940df09519f1`. Its matching fresh report is not present, so an actual motion launch remains gated. The older orientation report is not represented as current evidence.
- 1060 `preflight` validates the pinned input configuration, 14 static owners, and seven source capture pins. It reports `PASS_INPUTS_PINNED_NO_GEOMETRY_EXECUTED`, `native_steps: 0`, `physical_admission: false`, and `mass_mapping: NOT_RUN`. The large source captures themselves are not included.

## Validation

The publication rerun passed: 010b tests (13), 010c prelaunch tests (3), v16 observer-route tests (4 tests and 16 negative subtests), the v16 self-test, both runner syntax checks, and the 1060 source-only preflight. The expected 010b no-argument guard exits 64. The initial 010b package test exposed a missing 009 import closure; that failed log is preserved, the pinned closure is included, and the full suite passes on rerun. See `evidence/publication-rerun/receipt.json`.

The copied frozen owners retain their upstream bytes, including a few pre-existing end-of-file blank lines and one trailing space; these were not normalized because the files are hash-pinned.

## Limits

No native/GPU job, motion, geometry refresh execution, mass allocation, or physical qualification was run. The scripts retain Numi host paths and pinned upstream dependencies. A fresh matching orientation report must be produced before 010c can launch; this package does not satisfy that gate.
