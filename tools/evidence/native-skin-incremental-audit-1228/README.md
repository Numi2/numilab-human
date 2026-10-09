# Incremental exact skin-target audit

This bundle preserves the bounded CPU audit optimization and its actual-pose evidence. It is an offline algorithm/evidence task only: no geometry was adopted, no native run or GPU work was performed, and no simulation or anatomy qualification is claimed.

The helper compares baseline and candidate Float32 vertex bits, holds face row topology and target coverage fixed, reuses exact baseline target pairs only for unchanged skin faces, and runs the existing exact triangle predicate against every target for every changed face. It now requires four independently pinned identities: baseline world Float32 coordinates, skin face-index rows, all target geometry, and the canonical baseline exact-pair table. The caller must source the pair-table digest from retained original-scan evidence, not calculate a new expected digest from an untrusted pair table. Each per-target output marks its AABB work as changed_skin_faces_only.

## Actual-pose evidence

reports/comparison-001.json is the original full exact comparison at accepted step 155000. Its runner compared incremental and full exact candidate pair identities for all 859 target surfaces: 853 baseline pairs became 827 candidate pairs, with 0 added, 26 removed, and 827 unchanged across 14 changed skin face rows. Its top-level AABB totals were 79 fresh changed-face candidates versus 154,938 in the full scan. However, its per-target `changed_face_aabb_candidates` field is mislabeled and contains full-scan counts; do not use those per-target values. The raw comparison-001 report remains unchanged. The bounded synthetic fixture moved three packed skin vertices by 20 mm along a retained target face normal; it is not an anatomical candidate. Observed wall times were 39.57 s incremental and 68.00 s full-reference for that invocation; this is a single shared-host CPU audit measurement, not simulation performance qualification.


reports/comparison-003-final-c7-full-vs-incremental.json is a fresh full-reference comparison using the final `common_atlas_skin_clearance.py` source (SHA `c7c58cc902aaca199e86c32335936098c2ae4113155ec34e9d0f87aa337306be`) and the pinned exact predicate. It stores complete baseline, full-candidate, and incremental pair tables and checks exact pair identity for every one of 859 targets. Both candidate paths return 827 pairs; 0 were added, 26 removed, and 827 unchanged. The exact pair-table digests match. Correct per-target fresh-work counts sum to 79 and each row says `changed_skin_faces_only`; the full scan considered 154,938 AABB candidate pairs. This report corrects the per-target labeling problem in comparison-001 without editing that executed report. Wall times were 37.98 s incremental and 67.48 s full reference on a shared host; exclusivity was not established, so these are not isolated performance results.

Exact executed invocation: `cd /Users/n/numi-human-free-apex-two-family-1178 && PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python3.13 tools/evidence/native-skin-incremental-audit-1228/runners/compare_final_c7_full_vs_incremental.py`. The retained runner has a fixed report destination and refuses to overwrite it; a rerun requires a fresh runner copy/revision configured with a new output path.

reports/pinned-baseline-replay-002.json reruns the final guarded helper against the same accepted pose and the exact pair-table digest derived from the SHA-pinned retained witness stream, with all 859 target-summary counts checked against those witnesses. The replay returned 827 pairs, 0 added, 26 removed, 827 unchanged, and matched the prior full scan's candidate count for every target. The full candidate pair lists were not serialized in comparison-001, so replay-002 rechecks counts and identity guards; it does not claim to independently repeat the full pair-identity comparison.

The canonical pair-table SHA-256 is 4d43f9c63aa7e95951d27d4ca0765b3e3a0d00de7f690175d47d585e99b1cdf4. Its baseline world, skin topology, and target geometry identities are in the replay report. The accepted pack, receipt, original target summaries, exact witness stream, full-scan result, skin payload, and both executed reports/runners are pinned in external-artifacts.json.

## Source and tests

source/common_atlas_skin_clearance.guarded-ccb285.py is an exact reconstruction of the helper imported by replay-002; source/common_atlas_skin_clearance.final-c7c58.py is the final worktree source. The helper imported by comparison-001 had SHA `ded2138913d93be4e78af8ec8675e13f09f2191c3bbb3a5645add40354901ea3`; it was not recovered byte-exactly. A failed reverse reconstruction is recorded in provenance/source-version-availability.json, and no approximation is labeled as that original source.

The final focused test file is tests/test_common_atlas_skin_incremental_target_audit.py. The retained test log uses:

cd /Users/n/numi-human-free-apex-two-family-1178 && PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python3.13 -m pytest -q tests/test_common_atlas_skin_incremental_target_audit.py tests/test_common_atlas_skin_clearance.py

Result: 52 passed. git diff --check also passed. The helper change does not alter the exact predicate; it only limits fresh target AABB work to changed skin faces after fail-closed identity checks.
