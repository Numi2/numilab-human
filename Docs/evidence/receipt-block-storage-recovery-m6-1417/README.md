# Lossless receipt block recovery on SSH Numi
The pilot and remaining replacement phases recovered a measured 3,679,428,608 bytes (3.427 GiB). Free space after the second phase was 18,100,199,424 bytes (16.86 GiB). Background filesystem changes may affect this measured difference; allocated block estimates were not used as recovery claims.

Twenty retained receipt files had SHA256-equal blocks at the same offsets in another retained receipt. Each target was reconstructed from an independent APFS copy-on-write clone, rewriting only differing blocks. Each source hash and target full hash was verified before replacement; staged and final target bytes and metadata were also verified. Hard-linked targets or open handles are rejected/preserved. No source, receipt or evidence bytes were deleted.

Preserved metadata includes mode, owner, group, flags, ACL, extended attributes, birth time, and nanosecond access/modification times. Inode and ctime changes are recorded. Future modifications remain independent through copy-on-write.

The manifest, per-target verification journals, reports and exact scripts are retained here. The block-clone script reuses only the metadata functions from the pinned prior helper; it does not run the prior helper's duplicate-map replacement job.
