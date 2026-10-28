# Kind-aware D37 source preparation for candidate033

## Finding

`baseline-806.npz` stores a `node_kind` and a `node_source_row`. The source-row field is typed: for kind 0 it is a coarse representative row, while kinds 1 and 2 carry exact detailed-source IDs. The previous preparation treated the integer as a detailed D37 row without checking kind.

The concrete failure was canonical O vertex 16398, compact row 16264. Baseline806 node 1203 is kind 0 with coarse representative row 1233. Treating 1233 as a detail row selects donor 61 at a point 1.034101 m away. The unique P1 status-1 representative maps coarse row 1233 to detail binding support 24939, donor 23, whose current point is 0.762 µm away. That P1 row is binding support only; it is not topology ancestry. The emitted geometric registration is a separate D37 triangle face/barycentric projection, with residual recorded.

The corrected resolver preserves the baseline kind and coarse row, uses the P1 representative only to select the full 86-weight row, and keeps the geometry registration separate. Kinds 1/2 use exact detail IDs. Unresolved kinds remain unresolved for direct source identity and use explicit triangle geometry registration where available.

## Source and evidence map

- `tools/prepare_candidate033_d37.py`: kind-aware preparation tool copied from the working authoring source.
- `tools/test_prepare_candidate033_kind_aware.py`: regression for kind semantics, P1 coverage, donor agreement, and canonical row 16398.
- `canonical-007-preparation-receipt.json`: pinned source/output hashes, counts, replay checks, and admission fields.
- `source-provenance-crosswalk-summary.json`: compact mapping counts, input hashes, and the critical row trace.
- `worst-canonical-O-row-16398-trace.json`: original failure trace retained as diagnostic evidence.
- `source-rest-orientation-witnesses.csv` and `source-rest-orientation-witness-receipt.json`: 179 source-rest orientation reversals and their face-level witnesses.
- `canonical-generated-NumiHumanCandidateSkin033.hpp`: generated header hash-pinned by the canonical receipt.
- `regression-test.log`: actual regression output.
- `manifest.json`: SHA-256 values for the publication packet files and source code.

The full registered O node table, 86-weight table, and NHSKIN payload are not duplicated here. Their canonical hashes and paths are recorded in the preparation receipt and crosswalk summary.

## Verification

From the repository root on SSH Numi:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/numi/numi-human-prep-venv-20261005/bin/python tools/test_prepare_candidate033_kind_aware.py
```

Result: PASS. It reports 2,289 of 2,368 kind-0 rows with unique P1 representatives, 79 unresolved kind-0 rows, 6,386 exact kind-1/2 rows, 61,950 P1 IDs mapped bijectively to current D37 payload rows, and zero donor mismatches. The canonical row 16398 regression confirms the old >1 m interpretation and the corrected submicron support mapping.

The canonical preparation output is `prepared-007-kind-aware-canonical-compact`: 32,002 O vertices and 63,959 faces; NHSKIN payload SHA-256 `7a6c44b641b286497fbef73157e2945e2b7c4d5d2c423eff3b26590aa496abf4`. The compact node sidecar SHA-256 is `9e1fcdcb2d8812718a2cabd37908b4064553d89f8e3bab54a151afb75045b3cd`; the source transfer map SHA-256 is `817ce234b289dea3499c484185c613018861694c1c7abaf8b3a4c7c587d53b7f`.

The prepared current-reference Float32 orientation audit found 0 flipped or degenerate faces (minimum cosine 0.92003555). The separate source-rest unposed audit found 179 flips and 0 degenerates; the witnesses are retained here and are not hidden by the current-pose pass.

## Scope

This is source-preparation and reference-replay evidence only. The receipt records `admitted=false`, `native_runtime_verified=false`, `mass_ledger_pinned=false`, and `physical_motion_qualified=false`. It does not establish a positive child volume, conservative child mass mapping, or physical qualification. Prior failed attempts and source evidence remain at their original paths.
