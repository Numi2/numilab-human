# Quality volume archive chunk 002 proof

This source-only evidence package records the verified release and exact retirement of 12 historical quality-volume proposal NPZs. The archive is an external GitHub release asset; this repository package contains the index and receipts, not the array payloads.

## Verified archive

- Release: `human-quality-volume-archive-20261031-chunk-002`
- Asset: `quality-volume-archive-chunk-002.tar.zst`
- SHA-256: `51ed5f418bda9640a3204f997219c0416d50dccb13bbf0c9f647f6ff37e8e3ce`
- Size: `175571434` bytes
- Selected historical NPZs: 12 files, `282529137` original bytes
- Independent release download matched the GitHub asset digest. All 12 NPZ ZIP CRCs passed; 1,169 NPY array headers were read; one selected historical NPZ was restored and hash-checked.

The full plan and chronological receipts are under `payload/plan/`. `archive-index.json` raw file SHA-256 is `3086984bcfc99a936300c907563d82f113330266cc1d2845c0bf32f983dd0382`. The NPZ readback report records the SHA of the canonical JSON representation of that index (`daef380be10c58c20c0282ed38618b4e6c2bb098e8de185c120b57b9d1453bfb`); these are distinct, documented digest definitions.

## Evidence boundaries

This archive contains historical proposal inputs. It does not qualify anatomy or physics. Physical admission remains false, native steps are zero, and mass reconciliation is pending. The current-input proofs establish that the exact retired paths were disjoint from protected current inputs and that those protected inputs stayed unchanged.

See [RESTORE.md](RESTORE.md) for a hash-checked retrieval example.
