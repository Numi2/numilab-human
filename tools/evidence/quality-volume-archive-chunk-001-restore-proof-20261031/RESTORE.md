# Restore the archived historical volume proposals

Release: human-quality-volume-archive-20261031-chunk-001
Repository: Numi2/numilab-human
Exact uploaded asset name: quality-volume-archive-chunk-001.tar.zst
Release display label: quality-volume-proposals-792-794r001-795-chunk-001.tar.zst
Archive SHA-256: 9cb2e55a4bba2f5272aea543cdf62791650b0e0045fc2d453317b4e69bcb954d
Archive size: 176553369 bytes

The GitHub release asset is the retained backup. The original 12 local historical proposal arrays were retired only after the retained independent-download, archive-index, per-blob, CRC, header, and selected-restore checks passed. This publication contains the receipts and archive index, not the volume arrays themselves.

## Download and verify

    mkdir -p /tmp/quality-volume-archive-chunk-001
    gh release download human-quality-volume-archive-20261031-chunk-001 --repo Numi2/numilab-human --pattern quality-volume-archive-chunk-001.tar.zst --dir /tmp/quality-volume-archive-chunk-001
    shasum -a 256 /tmp/quality-volume-archive-chunk-001/quality-volume-archive-chunk-001.tar.zst

The SHA-256 output must equal 9cb2e55a4bba2f5272aea543cdf62791650b0e0045fc2d453317b4e69bcb954d.

## Restore one indexed member

The verifier is published at tools/evidence/connected-skin-fat-quality-trials-695-20261031i/restore_archive.py, source commit a7f784ecdd99f8b4ea70219304fff15b8d36cf85, SHA-256 dc9d4ffedf40e6ce39a78b0aa17b7bb905d5e26ea8953a571c5a5cdbc8bee12e. It streams and verifies every archive blob. This reproduces the independently tested all-quality-792 Fat proposal restore:

    python3 tools/evidence/connected-skin-fat-quality-trials-695-20261031i/restore_archive.py --archive /tmp/quality-volume-archive-chunk-001/quality-volume-archive-chunk-001.tar.zst --restore-root /tmp/quality-volume-archive-chunk-001-restore --restore-path /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/all-quality-792-root/round-01/proposal/fat-coupled-quality-point-proposal.npz
    shasum -a 256 /tmp/quality-volume-archive-chunk-001-restore/all-quality-792-root/round-01/proposal/fat-coupled-quality-point-proposal.npz

Expected restored member SHA-256: e91725a84af66a40bdb7395678df105615da1bfa983ac8d86676ef2f0a1661bc. To restore other entries, use their exact path from payload/plan/archive-index.json and pass one restore-path option for each. Use a separate restore root to avoid overwriting current project inputs.

## Retained verification and cleanup evidence

See payload/plan/archive-remote-verification.json, npz-readback-report.json, retirement-intent.json, retirement-receipt.json, current-input-proof-postretirement.json, local-copy-cleanup-intent.json, local-copy-cleanup-receipt.json, and lane-release-check.json. The initial archive-build-receipt.json records the earlier pre-upload stage; later receipts record independent verification and cleanup.

This is archive integrity and storage-cleanup evidence only. It makes no geometry, mass, physical, or native qualification claim.
