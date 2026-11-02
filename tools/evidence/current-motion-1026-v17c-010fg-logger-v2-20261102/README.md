# Direct-O motion verifier v17c, launch controls 010f/010g, and endpoint logger v2 patch

This evidence increment packages the current-source v17c verifier, the fail-closed first-run gate 010f, the increasing-horizon gate 010g, their control tests, and a minimal logger-v2 patch artifact. The Human repository does not contain the modified native app; the patch is isolated from the larger dirty R app file.

## Included controls

- v17c replays split reference-displacement high/low words for the accepted terminal FEM state and supports the full/compact report path. Its focused source tests passed 8 cases.
- 010f validates current input pins and requires a fresh passing signed role-orientation audit before creating a first-run job. Its prelaunch tests passed 5 cases.
- 010g allows the declared 2 to 16 to 128 to 1024-step progression, with each horizon initialized fresh from the same pinned baseline. Its horizon tests passed 15 cases.
- The logger patch changes only the endpoint capture version marker and adds initial-rest plus initial reference-displacement mode/high/low columns. A regression test applies this small patch to the exact R source base and compares the resulting file byte-for-byte to the expected patch-only transformation.

The exact source, test, and non-copied dependency hashes are in source-pins.json. The tests were run against the original H source locations on the pinned Numi host, not against extracted files in an unrelated environment. The package copies are byte-identical and individually hash-pinned.

## Validation

The v17c, 010f, 010g, logger patch, and package hash tests passed: 8, 5, 15, 1, and 1 case respectively (30 cases total). Python syntax checks and both runner shell syntax checks passed. See evidence/validation-receipt.json.

The native app build manifest is referenced by hash as logger provenance. That is a build record for the larger R source tree; it does not mean this isolated logger patch was run or physically qualified.

## Limits and gates

No native motion job, GPU execution, physical admission, or simulation qualification was performed for this package. 010f/010g are control sources and unit tests only. A fresh matching orientation audit report and all live run inputs are required at launch; they are not supplied here. The horizon stages are fresh initializations, not restart checkpoints.

The logger patch is stored as a review artifact only. It is based on the pinned R revision and does not include or modify the dirty R worktree. The larger R app source file contains unrelated work and is intentionally excluded.
