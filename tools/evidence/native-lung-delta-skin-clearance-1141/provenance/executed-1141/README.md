# Native skin-clearance delta transfer (1141)

This reader is prepared for a completed final native successor. It has not been run on a candidate and does not claim that the upcoming candidate is clear.

It transfers only the retained skin-clearance evidence from 1080/1115/1116 where transfer is proven. For a candidate it requires a completed 1120 transformed-lung delta report with exact accepted steps 0, 4991, 5375, 5759, 6111, 6495, 7743, 10000, strict current effective-area/config binding, successful verified runtime, unchanged tracked inputs, full self/cross pair coverage, exact row310 native copy evidence, and exact accepted body/respiration-state matches to 1116. It checks all non-target native primitives plus target/derived primitive identities, checks the captured skin XYZ and indices, and scans every changed triangle in rows 305-309 and 311 against all 109,211 captured skin triangles at all eight accepted poses. Row310 is covered only by the exact same-winding native-copy proof from 1120. The scan does not relax geometry predicates or turn remaining lung self/cross failures into a pass.

The retained baseline chain was checked by this reader on 2026-10-09. The check confirms the 1080 full baseline skin/self report, 1115 16-triangle-per-pose skin rerun, and exact 1113-to-1116 eight-pose bridge. See baseline-chain-check.json, source hashes in pins.json, and the 10 focused reader tests in test-log.txt.

Run only after root supplies a completed candidate run, candidate NHA and completed 1120 report. Replace every angle-bracket value with exact paths/hashes from that completed run; choose a fresh output path beneath the evidence root.

Command:
  /Users/n/numi-human-prep-venv-20261005/bin/python /Users/n/numi-human-resting-evidence-20261005/native-skin-clearance-delta-transfer-1141/skin_clearance_delta_1141.py --candidate-run <completed-native-run> --candidate-nha <candidate-resting-thorax.nhanatomy> --candidate-nha-sha256 <candidate-nha-sha256> --delta-report <completed-1120-report.json> --delta-report-sha256 <1120-report-sha256> --out /Users/n/numi-human-resting-evidence-20261005/native-skin-clearance-delta-transfer-1141/candidate-attempt-001

The program refuses an existing output directory. It writes skin-clearance-delta-transfer.json only after all eight poses are processed. A completed report with skin hits exits nonzero and preserves the intersections. Candidate preparation, native execution, and 1120 geometry analysis are separate authorities; this reader performs none of them.

## Limits

- This is discrete captured-pose skin-pair evidence. It gives no continuous-time clearance guarantee.
- A clear skin transfer is not a whole-anatomy qualification. Lung self-intersections, lung-lung crossings, cardiac interface issues, and physiology/endurance evidence remain separate.
- No new simulation or GPU work was performed for this reader.
