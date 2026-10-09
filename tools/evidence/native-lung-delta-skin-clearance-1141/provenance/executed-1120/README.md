# Exact delta audit for a same-state lung geometry successor

This CPU-only reader transfers the complete accepted-event ledger from the pinned 1096 full scan only when the successor preserves the exact native run identity needed by those events. It requires equal accepted body-state hashes, respiratory-state hashes, accepted times, verified loaded runtime, unchanged environment and numerical respiratory configuration. It also requires current receipt/config derivation to bind the candidate NHA payload and every lobe's geometry; candidate area binding is strict. The predecessor's historical geometry-only area mismatch allowance is not carried forward.

The supported delta keeps each audited row's vertex and face counts, stable ID, body/layer/flags, and index spaces. It permits position edits, regenerated normals (reported separately because they do not enter the XYZ intersection predicate), and fixed-row face index/diagonal changes for rows 305-309 and 311. Nonfinite position/normal data is rejected. Any edit to another NHA row except derived row 310, vertex reindexing, face insertion/deletion in an audited row, changed accepted state, asset/runtime mismatch, or invalid current map refuses transfer and requires a full scan. Row 310 is excluded as a duplicate surface from the six-owner intersection census. If it changes, the current receipt must hash-bind its unique exact same-winding face-to-lobe lineage; that lineage is recomputed against the selected NHA and checked against every actual accepted native pack using the existing 1115 topology/pleura checker. A missing lineage or any mismatched copied face refuses transfer.

For each of the eight accepted poses, unchanged predecessor intersections are reclassified against current exact records/maps, preserving real prior rejections. Every source- or pose-changed face star is then tested against all six audited rows for self intersections and all fifteen cross-row pairs with the existing exact predicate. New and removed intersections are therefore represented in the merged event ledger. Witness outputs are gzip JSONL; the summary reports full pair coverage and keeps unallowed/self and unclassified/cross events visible.

This is a discrete geometry delta audit. It does not establish continuous-time clearance, physiological plausibility, endurance acceptance, or biological validation. It performs no native simulation and does not change source assets.

The predecessor is fixed to `native-lung-transformed-pose-audit-runner-1096/attempt-001-native1113-geometry-only/report.json` (SHA-256 `740a9e704c97616f401ce767ef7253503846766da53277e0cefb22d83274800f`) and its predeclared input record (SHA-256 `d1c8107e391d3dfd2112f806afd36a0d528c1ee80b92c7e30fafb97874a43753`). The reader also pins the 1096 runner, the imported exact-predicate/current-map/lineage helpers, and the retained native row310-copy checker in code. No successor run has been analyzed by this package yet.

The child D-map adapter does not accept a fresh map merely because its report names the candidate. It requires the same immutable V8 D-map sidecar, a child composition report that hash-binds the V8 composition, V8 payload and map, and replays all 47,343 registered D/lobe triangles, source identities, winding, and reciprocal full-union boundaries against candidate rows. The V8 map itself is validated through its pinned 1078 parent lineage. Any changed registered triangle or row-index space refuses delta transfer. The current composition schema does not emit a child D-map count/area summary, so the adapter derives and validates those from the immutable map rather than treating absent report fields as proof.

## Invocation

Run only after a complete successor native trial and its current NHA, anatomy receipt, D311 map-composition report, and current lobe-lineage report are available:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python \
  /Users/n/numi-human-resting-evidence-20261005/native-lung-audit-exact-delta-1120/audit_delta_1120.py \
  --candidate-run /absolute/path/to/native-run \
  --candidate-nha /absolute/path/to/resting-thorax.nhanatomy \
  --candidate-nha-sha256 EXPECTED_NHA_SHA256 \
  --candidate-d-map-composition-report /absolute/path/to/current-D-map-composition.json \
  --candidate-lobe-lineage-report /absolute/path/to/current-lobe-lineage-report.json \
  --out /Users/n/numi-human-resting-evidence-20261005/native-lung-audit-exact-delta-1120/attempt-N
```

The output directory must be new and below the evidence root. Do not use geometry-only area mode. A failure found during an attempt is retained as an incomplete/failed attempt; a preflight refusal creates no output directory. Neither is grounds to relax exact predicates or copy a successful status from the predecessor.
