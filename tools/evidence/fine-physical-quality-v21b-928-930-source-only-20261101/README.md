# Fine physical quality v21b evidence through 930 (source-only)

This package continues the published 924-927 package (manifest SHA-256 `05fe7d9c9c350742e7e99f7253d462ed1c89782d329f963058e39678eb886a6a`). It contains compact reports, receipts, source scripts, and archive proofs. It does not bundle NPZ/NPY proposal arrays or archive assets.

## Selected source-only states

- **928:** 16 Skin / 10 Fat. Receipt, exact 806 audit, source proof 808, and independent v26b5b replay pass. The raw state retains an older proposal-pending status; the selected receipt and pinned replay are the qualification records.
- **929:** 15 Skin / 10 Fat. The selected `flips/receipt.json` records the independent fixed-boundary replay and source proof 808. The adjacent raw state still contains inherited proposal-pending fields; they are preserved verbatim and do not replace the selected receipt.
- **930:** 14 Skin / 10 Fat. The selected receipt binds inherited source proof 808 and external replay v4d. The external report keeps its own `PASS_EXTERNAL_INDEPENDENT_REPLAY_PENDING_FULL_CONTEXT` status; the package preserves that report alongside the overall selected receipt and does not rewrite either status.

All three receipts report `physical_admission=false`, `native_steps=0`, and mass reconciliation pending. The counts are source-quality results only.

## COW complete-payload check

The 926b receipt records a disposable scalar append to an actual 923 Fat NPZ clone: 167 arrays in/out, 166 unchanged compressed members verified, no changed existing keys, parent hash unchanged, and temporary clone deleted. The actual payload is not included. This checks the NPZ writer path, not anatomy or native simulation.

## Archived failed/unadmitted proposals

Archive chunks 020-022 retain their build/index/release/readback/restore/retirement records. Each chunk contains two proposal payloads. The recorded NPZ readbacks report passing ZIP CRCs and NPY headers; each includes an in-memory restore/readback receipt and a retirement receipt. Large `.tar.zst` assets remain on their referenced releases, not in this Git package. These were failed or unadmitted proposals and are not quality endpoints.

## Validation limits

This package validates receipt bindings, JSON parsing, and Python syntax. No geometry job, native build, solver run, or unit-test suite was run for publication. No claim of physical admission, native execution, or mass reconciliation is made.
