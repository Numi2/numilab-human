# Fine physical quality v21b bridge and P1 support evidence (source-only)

This increment follows the published 941–946 package (commit `e7db6256f39c049609b1fc4ad015a489bf844e9f`, manifest SHA-256 `fd1c471589fdaba04590fca246bc39eaba45c1f2a052f07ee70fbc2de547e48b`). It adds the current-fine 217 bridge 937 and current-role P1 support 938 sources/tests, the existing 943 source snapshot backup proof, and compact 947/948 failure reports. No NPZ/NPY arrays or archive payloads are bundled.

## Current selected source-qualified endpoint

The selected endpoint remains **943: 6 Skin / 6 Fat**, with the previously published 806 source audit and independent replay reporting zero unexpected crossings. Its receipt remains source-only: physical admission false, native steps 0, mass reconciliation pending. This package does not replace or extend that qualification.

## Added adapter and P1 support

- **Bridge 937:** includes the source adapter, fixture test, and fixture receipt (`PASS_SOURCE_ONLY_ADAPTER_CONTROLS`). The receipt explicitly does not claim source-cell ancestry, mass mapping, or TetGen execution.
- **P1 support 938:** includes the builder, owner/support code, and two focused test files. The retained shared-wrapper test receipt records 10 tests passing in 0.04 seconds, `py_compile`/`--help` checks as reported by the completed preflight, and no full-domain run. The tests do not establish mass reconciliation or physical admission.

## 943 backup and later failures

The existing source snapshot backup is linked as GitHub asset `quality-943-source-snapshot-20261101-001.tar.zst` (SHA-256 `b2b26b0ed82aaec6336f7498d48ca64d741e1da7a8510d6577fd1a240c89fde7`, 183,110,170 bytes). Its independent verification reports exact archive stream verification, all selected NPZ CRCs/NPY arrays loaded, and exact in-memory restoration. It also explicitly reports `full_original_ancestry_closure: false`. After that verification, only the local staging archive was removed; the receipt says all 29 original selected inputs were rehashed unchanged and no source/evidence file was removed. Archive assets are not bundled.

- **947:** no all-pass completion; zero operations; external source audit pending.
- **948 diagnostic and one-split attempts:** both report no valid coupled edge bisection. These are retained failures, not promoted candidates. No mutable 948 batch, 949, or 950 source/verifier code is included.

## Limits

All material remains source-only: physical admission false, native steps zero, mass reconciliation pending or not run. Included JSON and Python sources were parsed/syntax-compiled and copied-file hashes checked. No geometry solver or native build was run for this publication.
