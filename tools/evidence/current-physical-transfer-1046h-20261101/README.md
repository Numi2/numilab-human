# Current physical-transfer source package: 1046h

This compact source/evidence package is based on publication checkout Numi2/numilab-human at 2a70471338b4495e1f88b48a6ea7fd42f3cb738c (verified equal to origin/main before packaging). It preserves the existing 1020 geometry proof by reference; no candidate geometry or historical volume arrays are copied.

## Contents

- sources/1046h contains the exact 1046h producer and its direct code imports: 1046 base and the 149 tetrahedron projector.
- 1046h contains the small Skin/Fat NPZ and binary carrier packets, original producer receipts, independent readbacks, and gzip witness CSVs. witness-uncompressed-hash-manifest.json records each source CSV uncompressed SHA-256 and confirms decompression reproduces it byte-for-byte. producer-input-pins.json lists source/input hashes; large historical geometry and input arrays remain referenced, not copied.
- source-controls-006 contains the source tree and tests enumerated by the 1047 source-control manifest.
- quadrature-refinement-011 contains the 1047 frozen 011 source/control snapshot plus the applied publication copy of the helper, CLI, focused tests and validation receipt from commit 125e3913ad5856d9f2a3a60faa82603d0510e132; both source variants are kept distinct and hash-pinned.
- receipt-builder-013 contains the 013 receipt builder, test and source-control manifest.
- quality-fit-014 retains the unsuccessful continuous fit and its report. quality-fit-1018 retains the later LOCAL_ALL_GOOD_PENDING_FULL_REPLAY integer-F32 result and source; this is local constructive evidence, not full-source/native qualification.
- failed-trials/008 and failed-trials/009 preserve run specs, manifests, execution records and logs. Rejected outcomes remain failures.
- prior-1046-attempts preserves compact e/f/g source and status evidence. 1046e failed with KeyError distance; 1046f/g readback receipts report source-only pass and are explicitly marked superseded, not physical qualification.
- references.json points to the already-published 1020/808 source proof, checkpoint, and existing v14 motion verifier; the verifier is not duplicated.
- optimization-trials/014 and optimization-trials/015 contain their exact source snapshots, source-control manifests, focused test stdout/stderr, and preserved prior fixture failure evidence. Both manifests say PASS_SOURCE_CONTROLS; their shared pinned admission adapter is copied under optimization-trials/shared/. These are source tests, not mass or physical qualification.
- mass-mapping-011 preserves the timed run manifest, execution record, root termination note, log, and runner. It was stopped while minimum-density flow was still adding residual cuts.
- mass-mapping-012 contains only the current run manifest captured while RUNNING_NOT_ADMITTED; no live logs or partial outputs are copied.
- native-build-009 contains the compact build manifest and execution record. The binary was compiled but not executed; no native steps were run.


## Status

The 1046h carrier package and source controls remain source-only. Attempt 010 timed out after 600.015392834 seconds (exit 124) without a completed mass result; this is inconclusive, not infeasibility. Attempt 011 was root-terminated after 420.019936583 seconds (exit 241) with 95 residual-cut rounds while still in minimum-density flow; this is also inconclusive, not infeasibility. Attempt 012 is represented only by its run manifest at a RUNNING_NOT_ADMITTED snapshot, with native_steps=0 and physical_admission=false.

Optimization trials 014 and 015 report PASS_SOURCE_CONTROLS. Their focused tests and exact source snapshots are included, but no completed real mass result is claimed from those control passes. Native build 009 is compiled but unexecuted (native_steps=0). The selected 1020 source geometry remains unchanged and is referenced by the earlier source-only package. Native motion and physical admission remain unclaimed. No tests, compilers, geometry jobs, or native jobs were run to assemble this package.
