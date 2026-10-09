# Native lung delta audit and skin-clearance transfer (1120 + 1141)

## Published scope

This bundle publishes two CPU readers used sequentially after a completed native successor run:

1. The 1120 geometry-delta reader preserves the complete 1096 accepted-event ledger, reclassifies untouched events under the current map, and scans every changed-face star against the six lung/diaphragm owners at all eight captured accepted poses. It checks the current D311/lobe map, source edits, exact runtime/state identity, strict effective-area/config binding, and row310 copy lineage.
2. The 1141 skin-transfer reader checks exact baseline state identity and native primitive identity, verifies the captured full-skin XYZ/index stream, and scans every changed triangle in rows 305-309/311 against all 109,211 captured skin triangles at the same eight poses. It uses 1120's strict row310 exact same-winding copy proof rather than scanning the derived duplicate a second time.

These are discrete geometry readers. No final successor has yet been run through 1120 or 1141 in this publication. The included source-map smoke uses the earlier 1138 three-operation source candidate only; it is a 47,343-face source/interface-map bridge and does not establish native transformed-pose clearance. The baseline smoke validates retained 1080/1115/1113-to-1116 skin-clearance lineage only.

The prior full native 1096 scan reported 115 unclassified exact lung cross-intersection events. The 1120 reader keeps prior rejections visible and does not inherit a geometry pass from a source-map bridge. Its completed output may still report geometry failures. A clear 1141 skin scan is a separate skin-pair result and cannot turn the lung self/cross audit into a pass.

## Reproducibility on the Mac mini

The scripts use the pinned Mac mini evidence root at /Users/n/numi-human-resting-evidence-20261005 for hash-bound runs, source maps, and prior reports. The original executed source files and their hashes are retained under provenance/executed-1120 and provenance/executed-1141. Published source copies make only path-localization changes, recorded in provenance/source-provenance.json; audit logic and predicates are unchanged. The baseline data and historical run evidence are intentionally not duplicated here.

Run the focused tests from the repository root:

    /Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q tools/evidence/native-lung-delta-skin-clearance-1141/source/test_audit_delta_1120.py tools/evidence/native-lung-delta-skin-clearance-1141/source/test_skin_clearance_delta_1141.py

The packaged 1138 map smoke can be repeated to a new output directory. It verifies only the pinned source-level D311/lobe map bridge:

    /Users/n/numi-human-prep-venv-20261005/bin/python tools/evidence/native-lung-delta-skin-clearance-1141/source/verify_1138_source_bridge.py --out /Users/n/numi-human-delta-skin-transfer-publication-1141/tools/evidence/native-lung-delta-skin-clearance-1141/smoke/1138-source-map-bridge-attempt-002

The current smoke output is smoke/1138-source-map-bridge-package-002/source-map-bridge-report.json. It confirms 47,343 exact mapped triangles (305: 19,743; 306: 21,600; 307: 5,514; 308: 486; 309: 0) for the provisional 1138 source composition. It does not validate the later selected candidate, a captured native pose, area/config integration, skin, or physiology.

The baseline-chain check smoke is smoke/skin-baseline-chain-package-001.json. It reads the retained 1116 native run and confirms the frozen 1080, 1115 and 1113-to-1116 references for steps 0, 4,991, 5,375, 5,759, 6,111, 6,495, 7,743 and 10,000. No candidate transfer was executed.

## Running a future candidate

Wait for a completed native candidate, its current NHA and receipt, and a completed 1120 report whose current area/config status is strict pass. A complete 1120 geometry report may retain lung failures; those remain visible and do not qualify anatomy.

First run 1120 to a fresh output directory below the evidence root:

    /Users/n/numi-human-prep-venv-20261005/bin/python tools/evidence/native-lung-delta-skin-clearance-1141/source/audit_delta_1120.py --candidate-run <completed-native-run> --candidate-nha <candidate-resting-thorax.nhanatomy> --candidate-nha-sha256 <candidate-sha256> --candidate-d-map-composition-report <current-D-map-composition-report.json> --candidate-lobe-lineage-report <current-lobe-lineage-report.json> --out /Users/n/numi-human-resting-evidence-20261005/native-lung-audit-exact-delta-1120/attempt-N

Then run 1141 with that candidate, the same candidate NHA, and 1120's report.json and SHA-256:

    /Users/n/numi-human-prep-venv-20261005/bin/python tools/evidence/native-lung-delta-skin-clearance-1141/source/skin_clearance_delta_1141.py --candidate-run <completed-native-run> --candidate-nha <candidate-resting-thorax.nhanatomy> --candidate-nha-sha256 <candidate-sha256> --delta-report <1120-attempt-N/report.json> --delta-report-sha256 <1120-report-sha256> --out /Users/n/numi-human-resting-evidence-20261005/native-skin-clearance-delta-transfer-1141/candidate-attempt-N

Both tools refuse to overwrite prior output directories. They run CPU analysis only; neither launches native simulation or GPU work.

## Limits

- Accepted captures are discrete snapshots; no continuous-time clearance is established.
- The 1120 result is lung/diaphragm geometry only. Unclassified intersections, source-interface limitations, and any rejected geometry remain explicit.
- The 1141 result addresses skin contacts only and is reported separately.
- Neither tool establishes physiological plausibility, clinical validation, endurance, or whole-body all-pairs clearance.
