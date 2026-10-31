# Quality volume archive chunks 003–006

This source-only package records the archive plans, manifests, independent GitHub download verification, NPZ CRC/header readback, restore checks, and retirement receipts for four already-published release assets. The archive binaries are not duplicated here.

The chunk manifests identify each asset URL, SHA-256, size, selected file count, and restore tooling. Download an asset from its listed release URL, then use its corresponding `payload/archive-plans/chunk-NNN/archive-index.json` and `verify_npz_readback.py` / restore instructions.

Across the four chunks, 62 selected original NPZ files (1,340,114,779 bytes) were retired after the recorded independent-download and restore checks. The archive assets total 940,716,901 bytes. Each stored `archive-remote-verification.json` reports successful GitHub asset verification and restore/readback. The old direct-input checks are historical snapshots from those archive runs.

No native steps or physical admission are included or claimed.
