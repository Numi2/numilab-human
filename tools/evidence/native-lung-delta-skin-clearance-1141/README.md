# Native lung delta audit and skin-clearance transfer (1120 + 1141)

## Published scope

This bundle publishes two CPU readers used sequentially after a completed native successor run. The 1120 geometry-delta reader preserves the complete 1096 accepted-event ledger, reclassifies untouched events under the current map, and scans every changed-face star against the six lung/diaphragm owners at all eight captured accepted poses. It checks the current D311/lobe map, source edits, exact runtime/state identity, strict effective-area/config binding, and row310 copy lineage.

The 1141 skin-transfer reader checks exact baseline state identity and native primitive identity, verifies the captured full-skin XYZ/index stream, and scans every changed triangle in rows 305-309/311 against all 109,211 captured skin triangles at the same eight poses. It uses 1120's strict row310 exact same-winding copy proof rather than scanning the derived duplicate a second time.

## Actual 1159 result

The package readers were run on the completed 20-second native successor skin-927-lung-1159-viewer-018-v015-attempt1. The compact record is provenance/actual-1159/actual-1159-geometry-review.json; the full immutable reports remain in the Mac mini evidence tree at the paths and SHA-256 values recorded there.

The 1120 reader completed all six self-pairs and fifteen cross-pairs at accepted steps 0, 4991, 5375, 5759, 6111, 6495, 7743, and terminal 10000. Its status is complete_delta_scan_no_unclassified_hits: zero unallowed self-pairs, zero unclassified cross-pairs, and zero degenerate faces. Candidate area/config binding passed at 0.018687047064304352 m^2 with the same Float32 bits as its predecessor. The run's verified runtime, accepted body and respiratory states, input hashes, and derived row310 exact same-winding copy are bound by the full report.

The 1141 reader completed its eight-pose changed-face scan with status complete_changed_face_skin_scan_clear: 41 changed target triangles per pose against the exact 109,211-triangle captured skin stream, with zero exact skin intersection pairs. It also checked all 303,656 row310 triangles against their current lobe parents as exact same-winding native copies. This is a separate skin result and does not replace the 1120 checks.

Earlier failed attempts remain unmodified in the evidence tree. The compact review record lists hashes for the key failed runs; the retained attempt directories preserve their other logs and snapshots. Exact executed reader and test sources are copied under provenance/actual-1159; package source copies are independently listed in SHA256SUMS.txt.

## Reproducibility on the Mac mini

The scripts use the pinned Mac mini evidence root at /Users/n/numi-human-resting-evidence-20261005 for hash-bound runs, source maps, and prior reports. Earlier executed source snapshots remain under provenance/executed-1120 and provenance/executed-1141; exact 1159 executed files are retained under provenance/actual-1159. Published source copies make only path-localization changes recorded in provenance/source-provenance.json and are tested as packaged.

Run focused tests from the repository root:

    /Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q tools/evidence/native-lung-delta-skin-clearance-1141/source/test_audit_delta_1120.py tools/evidence/native-lung-delta-skin-clearance-1141/source/test_skin_clearance_delta_1141.py

The packaged 1138 map smoke can be repeated to a new output directory. It verifies only the pinned source-level D311/lobe map bridge:

    /Users/n/numi-human-prep-venv-20261005/bin/python tools/evidence/native-lung-delta-skin-clearance-1141/source/verify_1138_source_bridge.py --out /Users/n/numi-human-delta-skin-transfer-publication-1141/tools/evidence/native-lung-delta-skin-clearance-1141/smoke/1138-source-map-bridge-attempt-002

The source-map smoke is an earlier 1138 source composition, not the 1159 native audit. The retained baseline-chain smoke covers the 1080/1115/1113-to-1116 skin-clearance lineage only.

## Running a future candidate

Wait for a completed native candidate, its current NHA and receipt, and a completed 1120 report whose current area/config status is strict pass. A complete 1120 geometry report may retain lung failures; those remain visible and do not qualify anatomy.

First run 1120 to a fresh output directory below the evidence root:

    /Users/n/numi-human-prep-venv-20261005/bin/python tools/evidence/native-lung-delta-skin-clearance-1141/source/audit_delta_1120.py --candidate-run <completed-native-run> --candidate-nha <candidate-resting-thorax.nhanatomy> --candidate-nha-sha256 <candidate-sha256> --candidate-d-map-composition-report <current-D-map-composition-report.json> --candidate-lobe-lineage-report <current-lobe-lineage-report.json> --out /Users/n/numi-human-resting-evidence-20261005/native-lung-audit-exact-delta-1120/attempt-N

Then run 1141 with that candidate, the same candidate NHA, and 1120's report.json and SHA-256:

    /Users/n/numi-human-prep-venv-20261005/bin/python tools/evidence/native-lung-delta-skin-clearance-1141/source/skin_clearance_delta_1141.py --candidate-run <completed-native-run> --candidate-nha <candidate-resting-thorax.nhanatomy> --candidate-nha-sha256 <candidate-sha256> --delta-report <1120-attempt-N/report.json> --delta-report-sha256 <1120-report-sha256> --out /Users/n/numi-human-resting-evidence-20261005/native-skin-clearance-delta-transfer-1141/candidate-attempt-N

Both tools refuse to overwrite prior output directories. They run CPU analysis only; neither launches native simulation or GPU work.

## Limits

- The successful 1159 scan covers eight accepted snapshots over a 20-second preflight; it does not establish continuous-time clearance.
- A zero-unclassified scan means exact hits were accepted under the pinned source maps and adjacency rules; it does not mean that all surfaces are contact-free.
- The 1120 reader is limited to lung/diaphragm geometry, and 1141 separately evaluates skin contacts. Neither is whole-body all-pairs anatomy qualification.
- These reports do not establish physiological plausibility, clinical validation, biological validation, or endurance acceptance. The 310-second paired study remains a separate experiment.
