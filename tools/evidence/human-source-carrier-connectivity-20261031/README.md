# Source-carrier verification and fixed-boundary connectivity evidence

This compact source/evidence bundle combines the 188, 191, and 197 source-carrier checks with the 198–204 fixed-coordinate connectivity work. It contains code, compact receipts, and negative-attempt receipts. Large mesh proposals, original TetGen tables, full carrier-flow tables, and native binaries are not included; their hashes are retained in the receipts where available.

## Source-carrier scope

- **188:** verifies source-quadrature packet encoding, binary/NPZ parity, source moments, and scale-aware controls. The preserved receipt records 11 actual packet/negative controls. It is not a completed mass allocation.
- **191:** checks full-domain semantic-support permissions against the pinned partial flows. This is a partial-flow support check, not a complete allocation.
- **197:** validates truthful legacy-replay versus solver-free LP-feasibility receipt modes and recomputes composite carrier packet/array digests. Its 14 tests include a synthetic held-plus-LP build through the projection path. No real LP proposal or full allocation was available, so the full allocation verifier was not run. The optimizer remains `NOT_RUN`.
- The C++ finalizer fixture passed with the source/header hashes in `fixture-test-receipt.json`. The compiled binary is deliberately omitted. Its `ConservativeSpatialSourceCarrierQuadrature.hpp` dependency is pinned by SHA in the 197 test receipt and must be obtained from the matching NumiLab source revision.

## Connectivity scope

The 198–204 scripts and receipts record fixed-coordinate local tetrahedral connectivity changes. The independent replay confirms exact local F32 cavity volume, bit-identical coordinates, unchanged complete oriented boundaries, no duplicate tets, and no nonmanifold or misoriented internal faces. Across the final Skin and Fat receipts, 51,837 local operations are replayed.

The final candidates still fail global mesh-quality gates. Skin retains 22,814 normalized-determinant failures, 1,211 absolute-volume failures, and 23,147 union failures. Fat retains 49,525, 2,882, and 50,058 respectively. No new cell ancestry or mass mapping was produced. The source receipts therefore do not qualify a complete positive volume mesh or physical admission.

The 198 launch-preparation failure and 202 initial-attempt failure are retained alongside the later replay receipts. Intermediate proposal NPZs and the original `.node`/`.ele` inputs remain in the task evidence directory and are identified there by hashes; they are not copied into this Git evidence increment.

## Replay notes

The Python source snapshots are placed under `source/python/` and `source/connectivity/`. Their recorded runs used the task-owned `fine-physical-volume-706` inputs on Numi. The 188/191 actual-data test drivers create fixed output directories and were not rerun here; their existing receipts are preserved. The 197 synthetic tests were rerun with `/Users/numi/numi-human-prep-venv-20261005/bin/python`, and the C++ fixture binary was executed. The 198–204 verification scripts require the pinned handoff-142 mesh tables and the omitted proposal files.

A path-scoped `.gitattributes` entry and adjacent receipt preserve two frozen C++ source files with a pre-existing blank line at EOF; no global whitespace check is disabled and source bytes remain unchanged. The manifest lists each included payload path, SHA-256, and byte count. All receipts explicitly retain `physical_admission: false` and `native_steps: 0` where those fields apply.
