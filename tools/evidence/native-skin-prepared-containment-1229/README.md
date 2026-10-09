# Prepared exact point-location evidence

This bundle records a CPU-only exact-classifier optimization and read-only comparisons on retained accepted captures. It does not change target geometry or accepted native state, and it does not establish whole-skin clearance or anatomical qualification.

## Real-capture parity

reports/real-mesh-comparison.json compares legacy and prepared exact point-location results on accepted native capture step 0. The target is the left vastus lateralis (51005, 64): 6,690 captured faces in three closed components. The existing external-skin-enclosure-only rule selects its 6,666-face outer envelope for skin containment. All 38 complete result dictionaries matched: 12 target-boundary vertices, 24 captured skin vertices, one confirmed inside probe, and one far-outside probe. The comparison is bounded, not a full clearance scan.

Preparation took 0.745 s; 38 legacy queries took 1.154 s and prepared queries took 0.276 s. Tracemalloc reported 6,606,776 bytes retained and 7,018,124 bytes peak during prepared-record construction; recursive object size was estimated at 7,730,793 bytes.

## Integrated containment comparison

integration/full-skin-integration-comparison.json records an existing accepted capture at step 155000. It queried all 54,663 captured skin vertices against the same outer envelope. Legacy and prepared paths returned the identical empty inside-vertex set. The target had 6,690 faces total and 6,666 outer-component faces. Preparation took 3.008 s; legacy queries took 16.957 s and prepared queries took 4.285 s. These are sequential shared-host measurements, not isolated performance figures. The test did not run triangle intersection clearance or change the target's existing outer-envelope semantics.

## Tests and reproduction

The root-recorded focused test result is 73 passed, 1 deselected, and 8 subtests passed. The deselected source-geometry test requires absent Sources/partof_element_parts.txt; it is not counted as passed. See integration/root-integration-test-observation.json for the exact command and source hashes. That file is a transcription of the completed command result, not raw pytest stdout.

38-probe command:
PYTHONPATH=/Users/n/numi-human-free-apex-two-family-1178/src /Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/prepared-point-location-1228/real-mesh-comparison-001/compare_real_mesh.py

Full-skin containment command:
PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/prepared-point-location-1228/full-skin-integration-comparison-001/compare_full_skin.py

The retained runners refuse to overwrite their result paths. Large captures and source payloads are referenced by exact path, byte count, and SHA-256 in external-artifacts.json; no native pack, movie, or production binary is duplicated.

## Source snapshots

Predicate source SHA: 423180082e32d002375b612aad36f307ed439af07c04d655e6a8bedc87a338bd. The bounded 38-point probe used the prior common-owner snapshot SHA c7c58cc902aaca199e86c32335936098c2ae4113155ec34e9d0f87aa337306be. The integrated full-skin comparison used the current common-owner SHA cb249ec8fdc57ff95ff8bbbc3236b572eacb2aebee3bad083792bac39f0a67f4. These source phases are preserved separately.
