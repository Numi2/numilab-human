# Rejected NHSKIN bone-proximity reweight experiment (1185)

Prepared 2026-10-09 as a retention bundle outside any Git worktree. This package preserves the source experiment and its failure evidence; it does not publish or enable the inferred weight edit. No candidate `.nhskin`, distance `.npz`, pose-delta `.npz`, native pack, or binary was copied here. `external-binary-pins.json` records external source/asset paths and hashes separately from report-derived full-weight-field digests.

## Decision

The registered-bone proximity transform passed payload replay and compact-influence-cache checks, but the candidate is rejected for geometry. The complete 2026-10-09 CPU saved-pose screen evaluated only accepted step 47519 and only two variants: `compact-2.5x-full` and `compact-3x-full`. Both reduced the aggregate source-target pair count from 3,120 to 2,907 but introduced 1,575 new pairs, removed 1,788 prior pairs, and retained skin self-intersections (35 and 27 pairs respectively). The independent fold review details the 35-pair 2.5x case. These are CPU LBS displacement predictions added to the actual captured source Float32 skin positions; they are not candidate Metal captures and do not qualify all poses, support Jacobians, or physics. No full-cycle/native candidate qualification exists.

`compact-2.5x-half` passed the exact payload transformation gate, but its only saved-pose screen attempt (004) stopped after computing a partial target list and has a witness-serialization defect: exact rational intersection coordinates were cast to integers. Attempt 005 corrected the serialization to numerator/denominator strings, but screened only the two full-transfer variants; do not treat the half variant as geometry-screened.

## Build lineage

Build 007 and 008 use the same three distance/transfer parameter sets and the same registered distance array (SHA-256 `bf4c4405e45a89b5ecc3bc27308e9c5103f24187efb88a61808c547a25cc2e3f`). The builder scripts differ only in their output destination. Build 008 is the final checked helper version: it records the smooth bounded bilateral-support union `1 - product(1 - side_support)` and its payload gate verifies exact full Float32 weight replay, compact top-four cache consistency, exact coincident seams, unchanged geometry/bindings/topology, and near-unit row partition. Build 007's report did not hash-bind the helper source or declare the combination rule, so it remains a historical output pin rather than a fully source-closed replay. Candidate payloads from both builds are different bytes; all paths and hashes are in the external pin manifest.

The 008 independent payload gate is a necessary serialization check only. It does not validate posed geometry or justify admission. The two focused test modules in `source/tests` were run against the retained dirty worktree with `PYTHONPATH=src`; `focused-tests.txt` records **15 passed**. The source patch is retained for review/reconstruction only and was not committed or pushed.

## Failed screen sequence

- Attempt 001: private import of `cardiac_cavity_intersections` failed because its relative import lacked a package context.
- Attempt 002: screen call referenced an undefined `candidate`.
- Attempt 003: pair-difference accounting treated an integer count as an iterable.
- Attempt 004: got through all 859 target rows for the first `compact-2.5x-half` pass, then aborted on `SOURCE_SKIN_SHA` being undefined. Its partial pair records additionally serialized exact lattice points with `int(...)`, truncating rational coordinates. The partial outputs are preserved with the failure log.
- Attempt 005: fixed package imports, the argument/aggregation errors, exact-rational point serialization, source hash variable, source/receipt pin validation, and added exact self-pair records. It completed the one-pose screen for the two full-transfer variants, with the rejection counts above.

## Contents and limits

- `builds/`: exact 007/008 builders, build logs, reports, and payload-gate report. Binary outputs and the shared distance array are pinned externally.
- `screen/`: attempts 001–005, complete textual 005 result records, 004 partial result records, and 005 self-fold review. Generated `.npz` pose deltas are pinned externally.
- `source/`: base commit/worktree state, focused diff, regression test files, and actual 15-test result.
- `external-binary-pins.json`: external source/asset path and SHA inventory, plus distinct report-derived field digests; none of the external binary payloads are copied into this bundle.

This material is explicitly rejected diagnostic work. Do not use it as an active manifest, as a current skin asset, or as evidence of native or physical clearance.
