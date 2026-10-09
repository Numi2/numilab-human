# Exact skin audit index reuse

The skin clearance audit now builds one exact AABB tree for a candidate skin pose and reuses it across target surfaces. Target triangles query that tree; the original skin/target argument order is restored before the unchanged exact intersection predicate. Source face IDs, closed-box boundary contact, self-adjacency classification, and sorted pair reporting are preserved. The index snapshots immutable records and is discarded when the skin pose changes.

On accepted baseline step 47519 from the sealed 1170 run, all 859 targets, 176,771 AABB candidates, and 3,120 exact intersection pairs matched the original path and retained full audit, including every face-pair list. Both paths produced result SHA-256 207f3667d7ad08a52b30e7308bb068aee635a212025dd3e5a62b6c001ce04b3c. The existing common skin helper was independently exercised on the same complete inventory and produced the same result.

The measured scan changed from 82.918 to 63.075 seconds, a 1.315x improvement including target-record construction; audit work alone improved 1.500x. The integrated helper took 61.709 seconds and peaked at 655.9 MB process RSS. These are one-pose CPU audit measurements on the shared M4 Pro Mac mini while other work ran, not a native simulation speedup or an exclusive-host benchmark.

Validation: 48 targeted tests pass, covering the broad phase, exact cardiac predicates, and skin clearance. The initial test invocation had 20 passes and one missing-source-fixture error; the pinned tables and archive were copied from retained inputs and verified, after which all 21 predicate tests passed and the combined 48-test run passed. No source assets were added to Git.

The integration script completed its assertions but appended a literal backslash-n to its JSON output. The original bytes and executed script are retained. The valid integration-verification.json is a canonical serialization of the identical report emitted to integration.log; no audit was rerun or result value changed by this formatting correction.

This optimization does not clear any anatomical defect. The active skin fit retained its original source; a continuation can explicitly pin this helper after the bounded fit finishes. The native physical and physiological paths are unchanged.
