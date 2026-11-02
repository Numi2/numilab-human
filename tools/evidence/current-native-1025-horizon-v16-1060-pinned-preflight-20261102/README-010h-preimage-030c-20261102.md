# Source capture, preimage, and allocator diagnostics — 2026-11-02

This addendum records source/tooling controls and diagnostics. Its run receipts record `physical_admission: false` and `native_steps: 0`; no new native or GPU run was executed. It does not promote a failed allocator, a source-only packet, or a geometry diagnostic into physical qualification. The previous package README and receipt are unchanged.

## 010h sparse capture preparation

`horizon-010h/` adds a sampled-three capture option (initial, midpoint, terminal) alongside the existing six-sample roster. The 010h source control suite passed 24 tests; Python compilation and the shell syntax check passed. These are nonlaunch controls. They do not measure continuous swept geometry between samples or prove a native run.

The storage estimate described by the 010h documentation is derived from the pack-size formula and a declared boundary cap; it is not a measured runtime preflight. At the stated inputs, the four-copy estimate is 3,625,562,704 bytes, versus 4,734,121,996 bytes for seven copies. The runtime writer still checks actual pack sizes.

## Source002 composed-preimage diagnostic

`source002-preimage-audit-003/` is a read-only geometry comparison from the current 433 candidate to source002. The current mesh has 1,495,294 positive F32 tetrahedra. The attempted source002 preimage has 293,319 negative and 2,619 zero tetrahedra. The declared-parent replay disagrees for 1,011,078 child cells; 952,942 cells have a preimage vertex outside their declared parent, and 20,280 source parents have no child. The maximum current-candidate-to-source002 point residual is 15.7109 mm.

The separate 0.351375 mm measurement applies only to the source330-to-source002 leg. It is not the current-candidate displacement. These results reject the inferred preimage as a valid source mapping; the audit makes no source-registration, mass-transport, native, or physical-admission claim.

## 030c full-graph allocator diagnostic

`allocator-diagnostic-030c/` contains the CLI source, its ten trial-local headers, compile controls, run inputs, and compact full-cut metadata for the actual allocator attempt. The 32-entry compiler dependency manifest and `.d` file pin the full compile input closure; SDK and shared engine headers are hash-referenced rather than vendored. The full cut witness itself is hash-referenced and omitted from this repository because the gzip is 381,554,836 bytes (1,504,748,363 bytes uncompressed).

The allocator completed in 239.91 seconds and returned `INFEASIBLE_FULL_GRAPH_CUT` (exit 6): 12.822334153 kg sent against a 12.998725017 kg minimum, leaving a 0.176390865 kg deficit. This is 0.033068034 kg less deficit than the earlier 027 full-graph cut (0.209458899 kg), but the graph remains infeasible and no candidate is promoted.

## 031h proposal, 032d packet, and 034 readback

`adaptive-joint-032d/` preserves the 031h adaptive-anchor source and report, the 032d proposal source/report, and the 034 packet bridge/readback records. The 031h proposal met 13 of 15 historical cut constraints. The 032d report records 570 adaptive proposal rows and passing nonnegative-weight and mass/moment checks, but its post-restore/blend gate did not meet all 26 cut constraints. Current027 component 00 and 03 remain short by 0.029072190 kg and 0.004762126 kg. Other historical component constraints also remain short.

The 034 bridge passed source-only packet readback. It did not run the allocator. The large packet CSV, NPZ, and BIN files are excluded; their sizes and hashes remain in the pinned receipts. Neither the proposal nor the packet readback establishes a feasible full flow or native/physical admission.

## Evidence map

- `horizon-010h/focused-checks-001/receipt.json` records the 24-test source-control result.
- `source002-preimage-audit-003/run-001/report.json` records the read-only preimage result.
- `allocator-diagnostic-030c/actual-001/execution.json` and `capacity-witness/manifest.json` record the full-graph infeasibility and omitted witness hash.
- `adaptive-joint-032d/032d-proposal/joint-lp-report.json` records the unmet constraints; `034-packet-readback/batch-packet-verification.json` records source-only readback.
- `evidence/sparse-capture-preimage-001/package-rerun-001/` records package-local focused checks.

Copied upstream sources remain byte-identical to their pinned SHA-256 values. The staged whitespace check reported only trailing blank lines at EOF in eight frozen source files; they are retained to preserve source identity. No other whitespace findings were reported.

All source versions, outputs, and compact receipts are pinned in `addendum-manifest.json`. `SHA256SUMS.txt` covers the files listed there plus the manifest and this addendum README, excluding `SHA256SUMS.txt` itself.
