# Fine physical volume quality evidence: v17 and separate 795/799/806 branches

This package contains source-only mesh and tetrahedral quality evidence. It is not physical admission, a native run, or material-mass reconciliation.

## Results

- **Row-scaled v17 (807), two rounds:** final verified counts are Skin 53 and Fat 214 native-quality failures; exact surface crossings are 0. Both rounds have independent source/F32 replay receipts. The recorded LP trace reports HiGHS optimal termination. The final state marks boundary source binding `PENDING_METADATA_REBIND_NO_GEOMETRY_CHANGE`; mass reconciliation remains pending.
- **Complete-star 795:** independently promoted source-only state is Skin 57 / Fat 221. Quality defects remain; mass mapping was not run.
- **799:** latest verified round 2 is Skin 57 / Fat 219, source-only. Round 3 was not promoted and is excluded.
- **806 partial-cone alternative:** verified state is Skin 55 / Fat 212, source-only, based on 799 round 2. This is a separate alternative branch; its counts are marked `DO_NOT_SUM` with 799 or v17.

All included branch receipts have `physical_admission=false` and `native_steps=0`. Per-material volume changes may be reported, but mass conservation is not claimed. The stopped 806 native attempt produced no output pack; no native output or overlay is included. External NPZ files are excluded; `payload/external-npz-pins.json` retains source-recorded hashes and references.

## Provenance

Scripts and reports are copied byte-for-byte from their retained source paths. Shared dependencies already published in the predecessor manifest are not duplicated. See `manifest.json` for file hashes, the upstream manifest pin, and branch inventory.
