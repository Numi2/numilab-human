# SSH Numi checkout closure — 6 November 2026

Archived available work in `Numi2/numilab-human` and ended this session. This is a recovery checkpoint, not a feature release or a physics/performance qualification. No simulation, build or experiment was resumed.

The inventory covers **52 distinct registered locations**, including **22 dirty checkouts** and **0 missing temporary registrations**. Across the three owning repositories there are 33 distinct dirty checkouts and six absent temporary registrations. Existing local working trees, deletions and staging state remain intact; unvalidated changes were not merged into `main`.

The full Human deliverable and its performance/motion qualification remain unfinished; see the [Human session handoff](session-close-20261106/README.md).

## Preserved and checked

- [Release and recovery payloads](https://github.com/Numi2/numilab-human/releases/tag/ssh-numi-checkpoints-20261106): dirty/untracked files, exact patches/index state, source references, available reflog/stash history and unpublished Git objects where applicable.
- [Compact recovery index](SSH_NUMI_CHECKOUT_CLOSURE_20261106.json): archive SHA-256 values, member inventory and independently downloaded restore-verification receipts.
- [373 changed/untracked file paths](https://github.com/Numi2/numilab-human/releases/download/ssh-numi-checkpoints-20261106/numilab-human-ssh-numi-20261106-index.tar.zst) are indexed for this owner. Shared content reuses verified archived copies.

Every new payload archive was downloaded independently, decompressed and restored member-by-member with size/SHA-256 checks. Unpublished Git bundles were independently unpacked and their required new objects checked. Original source heads, refs, index entries, status, files and symlink targets were rechecked before this documentation-only publication.

## Recovery and limits

Download the index archive and its manifest from the release, verify the listed SHA-256, and decompress with `zstd -d --long=29`. Its inventory maps original paths to payload archive members. Restore into a separate directory. Fetch release tag `ssh-numi-checkpoints-20261106` from this repository before verifying/unbundling source bundles; its published history supplies their prerequisites. Retain recorded patches/index entries when reconstructing staged and unstaged state.

Files already absent from missing temporary directories cannot be reconstructed as untracked working state; available registered commit heads/history are preserved. Archived queues and commands are evidence and must not be launched automatically. Local sources were retained and remain dirty where observed. No cleanup/reset or qualification claim is implied.
