# Native transformed-pose lung audit 1096

## Result

The eight-pose geometry scan completed, but it **did not pass geometry acceptance**. It found **115 exact cross-surface intersection pairs that remain unclassified**. All six self-pairs and fifteen cross-pairs were scanned at each of eight accepted snapshots (168 pair checks); there were no unallowed self-intersections, no degenerate faces, and no missing pair coverage. Exact intersections were retained without a distance tolerance or automatic contact waiver.

The 115 unresolved cross-pair events are distributed as follows:

| Accepted step | Unclassified events | Pair detail |
| ---: | ---: | --- |
| 0 | 0 | — |
| 4,991 | 33 | 305–308: 25; 306–309: 4; 307–309: 4 |
| 5,375 | 19 | 305–308: 15; 307–309: 4 |
| 5,759 | 12 | 305–308: 8; 307–309: 4 |
| 6,111 | 0 | — |
| 6,495 | 30 | 305–308: 26; 306–309: 4 |
| 7,743 | 6 | 305–308: 6 |
| 10,000 | 15 | 305–308: 11; 307–309: 4 |
| **Total** | **115** | **305–308: 91; 306–309: 8; 307–309: 16** |

The report therefore records `geometry_scan_status=FAIL_or_incomplete`. “Unclassified” means the exact event did not match a declared reciprocal-face or shared-feature rule; this scan does not claim that all 115 events are independently established material defects.

There is a separate integration failure. The selected run's respiration configuration has effective diaphragm area **0.018687047064304352 m²**, equal to both the candidate-recomputed scalar and the prior derivation scalar. However, the configuration/derivation binds payload SHA `7f6a8175…c414e92`, while the selected NHA payload is `1b62f569…6747b8bc`; the per-lobe geometry hashes also differ. The report records `physiology_integration_status=FAIL_area_binding_mismatch` for `payload_sha256` and `per_lobe_geometry_sha256`. This is a stale source-geometry binding, not a different numeric area. The run must not be represented as integrated anatomy/physiology acceptance.

## Scope and inputs

This is a read-only, CPU exact-predicate census over eight saved native accepted poses from run 1113: steps 0, 4,991, 5,375, 5,759, 6,111, 6,495, 7,743, and the true terminal step 10,000. Requested duration was nominally 20 seconds; the accepted clock uses Float32 `dt=0.0020000000949949026 s`, so the terminal receipt time is 20.000000949949026 s. Each pose includes six self-pairs and all fifteen cross-pairs among lung/pleura rows 305–309 and diaphragm row 311.

The selected NHA SHA-256 is `1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc`. The run and source maps are bound in [the declaration](result/declaration.json), [the preflight](preflight-native1113-001.json), and [the input manifest](input-manifest.json). The full exact raw-witness streams remain at the pinned evidence path listed in the declaration; this repository bundle includes the compact filtered event rows for the six nonzero poses, not the large gzip streams. The retained scan report SHA-256 is 740a9e704c97616f401ce767ef7253503846766da53277e0cefb22d83274800f.

This is discrete accepted-pose evidence only. It does not establish continuous-time clearance, functional physiology, or clinical validity. No geometry or physics source was changed by this audit.

## Reproduction and tests

The pinned Mac mini evidence directory and Python environment are required; the large source/run inputs are not duplicated in this repository bundle. The original invocation is documented in [the retained runner README](source/README-1096.md). To reproduce, first verify its input hashes and choose a fresh output directory; never overwrite the retained attempt.

Focused map/lineage tests passed 17 tests in the retained run. The packaged test copies resolve the tested runner and adapters from this directory while retaining the same test assertions. Run on the pinned Mac mini with:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python3.13 -m pytest -q tools/evidence/native-lung-transformed-pose-audit-1096/tests
```

The execution log is [test-suite-package-001.log](tests/test-suite-package-001.log); the historical original test log is retained alongside it. Exact pre-relocation test sources are preserved under [provenance](provenance/); the packaged test copies change only the module-root paths so they import this bundle's source copies. Check bundle integrity with `shasum -a 256 -c tools/evidence/native-lung-transformed-pose-audit-1096/SHA256SUMS`.
