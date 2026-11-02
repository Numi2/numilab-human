# Arbitrary carrier-point owner control 031

This source-only package preserves the 031 native-owner fixture for arbitrary source-carrier points. It exercises four noncanonical interior barycentric sites per role, first-moment preservation, and the current carrier binder/validator.

## Recorded controls

The pinned owner run reports all six controls passing: arbitrary projection accepted; wrong first moments rejected by the native owner; canonical mode still rejects noncanonical sites; outside-tet sites, over-radius sites, and wrong donor tuples are rejected. The exact compiler invocation, return codes, external include paths, source hashes, and control output are recorded in `controls/source-control-031.json`. The original stdout is preserved beside it.

## Dependencies and limits

The frozen 029 support header, carrier-owner header, and canonical fixture source are pinned in the control manifest by path and SHA-256; they are not copied into this package. The compiled binary is also referenced by pin only and is not included. This is a focused owner control: the full allocator was not run (`NOT_RUN`), there were no native steps, and it makes no physical-admission or human-qualification claim.
