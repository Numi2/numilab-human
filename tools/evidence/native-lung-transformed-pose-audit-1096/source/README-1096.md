# Native lung transformed-pose audit runner 1096

This fresh, read-only CPU geometry-audit runner adapts the retained 1042/1084/1085 exact predicates for the selected v8 D311 map and v2 reciprocal lobe-lobe bridge. It does not rewrite anatomy, physics, captures, or registered study inputs.

The selected native run is the completed 20-second 1113 capture. It has eight accepted snapshots, including accepted step 10000. The requested nominal horizon is 20 seconds; the native Float32 clock is reported from owner receipts.

The v8 map binds 47,343 reciprocal D311/lobe faces: row305=19,743; row306=21,600; row307=5,514; row308=486; row309=0. The v2 lobe map declares all ten pairs, including explicit zero pairs. The reader checks current Float32 triangles, source parent rows, opposite winding, and mapped-union boundaries before scanning.

## Important limitation

The native invocation uses the frozen 1078 respiratory configuration. Its scalar diaphragm area is still 0.018687047064304352 m², but its provenance and lobe-geometry hashes bind to the 1078 payload, not to the selected v8 NHA. The selected 306–309 lobe rows differ from 1078. Therefore the default integrated area-binding gate correctly refuses this run. The explicit --geometry-only-area-mismatch switch allows only an exact discrete geometry census to proceed; the report must record the failed area binding and cannot be treated as integrated anatomy/physiology acceptance.

This scan reports the six lobe/diaphragm self-pairs and all fifteen cross-pairs at each of the eight accepted poses. It retains all raw exact witnesses in gzip level 1; no distance tolerance or automatic contact waiver is added. It is not continuous-time, functional, or clinical qualification.

## Validation performed before the scan

- Python 3.13 compile check passed.
- The focused v2 lineage and current v8 D-map adapter suite passed 17 tests (test-suite-003.log).
- preflight-native1113-001.json validates the completed1113 owner run, D map, lobe lineage, stale area binding, and 2-worker memory estimate. It records steps 0, 4991, 5375, 5759, 6111, 6495, 7743, 10000 and 95 tracked input hashes.
- The resource probe is the pinned 1090 one-pose 1078 run. It measured 2.013 GB peak RSS at 582,798 triangles. Current1113 has 582,792 triangles; the conservative estimate is 3.524 GB per worker including 75% reserve. Two workers estimate 7.047 GB, below the 12 GiB gate.

## Reproduction command

Run only after confirming the attempt-001-native1113-geometry-only output directory does not exist:

/Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-resting-evidence-20261005/native-lung-transformed-pose-audit-runner-1096/audit_lung_cycle_1096.py
  --run /Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-1113-viewer-018-v015-attempt1/native-run
  --out /Users/n/numi-human-resting-evidence-20261005/native-lung-transformed-pose-audit-runner-1096/attempt-001-native1113-geometry-only
  --nha /Users/n/numi-human-resting-evidence-20261005/native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8/final/resting-thorax.nhanatomy
  --nha-sha256 1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc
  --d-map-composition-report /Users/n/numi-human-resting-evidence-20261005/native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8/composition-report.json
  --lobe-lineage-report /Users/n/numi-human-resting-evidence-20261005/native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8/current-reciprocal-map-report-v2.json
  --geometry-only-area-mismatch
  --workers 2
  --probe-report /Users/n/numi-human-resting-evidence-20261005/native-lung-transformed-pose-audit-runner-1090/attempt-002-step0-1078/report.json

input-manifest.json pins the reader, tests, source adapters, selected geometry/map/receipt inputs, native invocation/metadata/log, accepted captures, resource-probe records, and owner/parser/predicate dependencies. The final declaration/report separately pin the actual scan inputs and results.
