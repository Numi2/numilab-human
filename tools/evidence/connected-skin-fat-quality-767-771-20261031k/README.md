# Partial cavity quality replay: 763, 767, and 772 round 001

This package preserves the 750/767 bounded cavity producers, kernel-cone helper 746, authenticated external promoters 761/768/771, their reports and readbacks, and the 772 round-001 independent replay. Large source and proposal NPZ arrays remain external and are pinned by path and SHA in `manifest.json`; no fat volume arrays are bundled.

## Verified results

- 763 plus promoter 761: the full bounded report is `PASS_WITH_PARTIAL_NATIVE_DEFECTS`; the external replay and fresh typed source-volume crosswalk pass. Skin improves 72 to 69 native-quality failures; Fat improves 403 to 336.
- 767 plus promoter 768: the source crosswalk and independent replay pass. Skin is an inherited no-op at 69; Fat improves 333 to 275.
- 772 round 001 plus promoter 771: independent source-boundary replay passes. Skin remains 69 as an inherited no-op; Fat improves 275 to 269. Verified-state SHA is `a6914b209d046aa49cee8e1e71717bb2a784b9e06d1e310cfe4ea83a8c81b0c3`.
- Round 002 is retained as `NO_VALID_BOUNDED_PROGRESS`; round 001 remains the latest verified state.
- The tiny 771 control fixture rejects four altered-proposal mutations and rejects promotion without source proof. This is a synthetic control only.

All actual result receipts record `physical_admission: false` and `native_steps: 0`. Native-quality failures remain; no physical admission, mass allocation, or clinical qualification is established. This package excludes the experimental 756/742/v15 branches.
