# Exact Float32 skin-target audit broadphase

This increment accelerates offline exact target-intersection auditing. A fit-owned cache remembers only a digest after the owner completes full exact validation of all target geometry for that pose. On a cache hit, a conservative Float32 AABB prefilter chooses target rows potentially overlapping changed skin faces; the existing exact integer-lattice predicate still decides intersections. The output preserves every target row and original target-face identity. Calls without the private cache keep the legacy full-validation behavior.

## Exact-pair evidence

`reports/full-cache-comparison-001.json` and `full-cache-comparison-002.json` are retained all-target parity comparisons at accepted steps 0 and 155000. For each, full scan, first cache-miss validation, and cache-hit broadphase had identical exact pair lists over all 859 target surfaces. These runs used intermediate working-tree snapshots, with exact source/diff hashes embedded in each report (`5d0865f7…` and `8e2144bc…`); they predate the final per-target packed-Float32 reuse adjustment. They are parity evidence for those source revisions, not fresh full-scans of the final source file.

`reports/final-owner-sparse-incremental-profile.json` is the successful sparse incremental path using the final owner source SHA `ae9f4c8a18434ecce71a61472ac4732845cd979c2281c5f2b57a73ef7a5758d2`. At accepted step 0 it matched all 859 pair tables against the retained exact sidecar. Uncached full target validation took 29.04 s; the same-geometry cache-hit incremental audit took 5.60 s. It constructed 4,891 exact target records on the hit versus 4,801,596 for the full validation, and ran 11,110 candidate-pair predicates. The process peak RSS was about 1.82 GB while forward reconstruction and three audit variants were in flight. This is one offline CPU profile on a shared host, not native simulation RTF or a general performance guarantee.

The final sparse report checks pair-table digest and per-target counts against the retained accepted-pose evidence. The full-pair identity fixtures were run before the final packed-position micro-optimization; no new 859-surface full-pair scan was repeated after that last change. There was no fit execution, candidate adoption, native run, or GPU work.

Two failed incremental-profile attempts are preserved in `provenance/`. Attempt 001 stopped before an audit because its adapter imported a module without the newly threaded private cache keyword; its source file is an explicitly labeled reconstruction from the post-failure edit, not an exact executed-source snapshot. Attempt 002 completed the uncached, miss, and hit pair comparisons, then failed in profiler-only memory reporting on an empty sample list. The successful attempt is separately pinned in `runners/profile_incremental_cache-final.py` and the sparse profile report.

## Validation

Command:

```sh
cd /Users/n/numi-human-target-broadphase-1241
PYTHONPATH="$PWD/src" /Users/n/numi-human-prep-venv-20261005/bin/python3.13 -m pytest -q tests/test_common_atlas_skin_clearance.py tests/test_common_atlas_skin_incremental_target_audit.py
```

Result: **61 passed**. `git diff --check` also passed. The tests cover exact cached/full equivalence, sparse changed-face checks, source and topology bindings, canonical Float32 behavior, and invalid inputs.

## Required fitter integration

A fitter must create a fresh private validation set for each fit, warm it only by calling the owner’s full exact validation for each pose’s target geometry, then pass it to incremental checks for that same geometry. The pose-specific geometry hash prevents reuse across changed target geometry. Do not populate this cache from a pair table or a reported success boolean.
