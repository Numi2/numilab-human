# Quality 695 failed-trial archive

This package preserves a bounded set of historical, unadmitted trial files from all-quality-695-root. It is not the live checkpoint and does not establish physical admission.

## Contents and status

The archive contains 40 unique NPZ blobs (637,509,229 source bytes): 24 Skin/Fat proposal outputs from trial01 rounds 001–012, plus 9 D and 7 O input snapshots that the read-only consumer audit cleared for retirement. Every proposal report recorded FAIL_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185, physical_admission: false, and native_steps: 0. Trial JSON state/config/report/telemetry remains at its source path. Current checkpoint and current-state ancestry inputs were excluded from retirement and are recorded in the selection plan and index.

The archive has a content-addressed blobs/<sha256> layout. archive-index.json maps every selected original path to its SHA-256, byte count, and archive member. checkpoint-snapshot.json is the frozen small checkpoint pointer captured for this archive; it reported 730 as its checkpoint current state. The contemporaneous best state 738 and its ancestry are separately pinned. The moving live checkpoint is not used as a mutable verification input.

## Verify and restore

Run the verifier to stream-decode every blob, compare every content hash and size with the index, and optionally restore one or more selected files under a chosen root:

    python3 restore_archive.py --archive quality695-trial01-failed-proposals-001-012.tar.zst
    python3 restore_archive.py --archive quality695-trial01-failed-proposals-001-012.tar.zst --restore-root /tmp/quality695-restore --restore-path /original/absolute/path/to/one.npz

The selected original files were retired only after the GitHub release asset was independently downloaded, its SHA-256 was compared with both the local archive and GitHub asset digest, all content-addressed members were streamed and checked, and one selected NPZ was restored and rehashed. The retirement receipt and tombstone list each removed source path and restore mapping. JSON evidence, unselected inputs, current sources, and current DAG artifacts remain untouched.

## Claim limits

This archive contains failed or intermediate trial artifacts. It makes no claim of all185 acceptance, physical qualification, native solver steps, mass allocation, or a clinical result.

