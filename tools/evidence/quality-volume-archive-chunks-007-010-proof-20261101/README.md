# Quality volume archive chunks 007–010

This package preserves the compact selection plans, archive indexes, independent-download receipts, NPZ CRC/header readbacks, selected-file restore checks, retirement proofs, and restore scripts for chunks 007–010. Archive payloads remain in their GitHub releases and are not duplicated here.

The four chunks record 56 selected file references, 54 unique NPZ blobs, and 1,297,655,250 bytes of selected originals. Their compressed assets total 815,479,273 bytes. Live GitHub tag/asset ids, upload state, sizes, and SHA-256 digests were checked during preparation.

The index references differ across receipts: the current build receipt/index SHA differs from the NPZ readback report SHA for each chunk. For 007 the remote verifier matches the build index; for 008–010 the remote verifier and readback report match each other, but not the current build index. The package retains every value and makes no claim that these hashes are identical.

Chunk 007’s initial verifier attempt stopped on a missing metadata key; its erratum records that no source files were retired during that attempt. The corrected verifier, restore proof, and post-retirement receipt are retained. Chunk 010’s erratum corrects a previously misreported asset id that belonged to chunk 009; the corrected id and digest match current GitHub metadata.

Direct-input checks are historical archive-time snapshots. They do not imply current geometry or physical qualification. No native steps or physical admission are included or claimed.
