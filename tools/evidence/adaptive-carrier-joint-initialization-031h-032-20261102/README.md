# Adaptive carrier joint initialization 031h / 032 — source-only result package

**Physical admission: false. Native steps: 0. The 031h/032 proposal remains unselected.** Full-flow run 029 produced a 0.9721693390 kg deficit, worse than the earlier actual-027 full-graph deficit of 0.2094588993 kg by 0.7627104397 kg.

This package preserves the 031h adaptive preparation/LP, the corrected 032 packet bridge, passing packet readback, and a completed but infeasible full-flow attempt. The source controls and packet readback pass; the actual full minimum-flow allocator did **not** meet required density.

## 031h preparation and LP

The frozen 031h producer generated adaptive source-tet sites and an adaptive slack LP proposal. Its source-only report records solver success and replayed parent-moment checks, with 13 of 15 cuts meeting demand. Two cuts remain short: `actual025_component_00` by 0.026278605918986148 kg and `actual025_component_03` by 0.005055771702920972 kg. No native steps were taken; physical admission is false. Proposal and adaptive-site NPZ arrays are referenced by hash but omitted.

## Corrected 032 packet bridge

The corrected 033 bridge convex-blends a 1e-6 portion of the original positive source quadrature into the LP quadrature, preserving per-parent mass and first moments while restoring prior support. It rejects nonfinite witness distances and binds the matrix run-manifest and execution hashes. Its self-test and in-memory syntax compile pass.

Run-004 packet readback is `PASS_SOURCE_ONLY_ADAPTIVE_PACKET_READBACK`. It restores 56,564 zero-LP anchors and records per-role packet receipts and verifications. Large BIN/NPZ payloads and witness CSVs are omitted; their paths and hashes are listed in `package-pins.json`. Packet readback is not allocator-flow qualification.

## Actual full-flow result

Actual-029 baseline-route run-001 completed 20 minimum-flow BFS phases and 1,692,234 augmentations, then exited 6 as `INFEASIBLE_MINIMUM_FLOW`. It routed 12.026555678552887 kg against a 12.998725017533202 kg minimum, leaving a 0.972169338980315 kg deficit. The attempt ran 20 BFS phases, 1,692,234 augmentations, and 1,032,849 edge additions; its terminal cut summary examined 499,114,596 pairs. The capacity-witness CSV failed to write and no verified-flow output was emitted. Native steps were zero and physical admission is false.

The preserved `run.log` is a merged stdout/stderr stream. Its capture manifest records that chunks 1–7 were reconstructed from exact tool-returned output text and chunks 8–9 were captured directly from the active session. This package does not label the reconstructed portion as direct native stdout.

## Failed attempts and limits

`failures/failure-index.json` preserves compact preparation errors, four bridge failures, the child-196 preflow rejection, and this completed infeasible flow attempt with exact source, manifest, execution, and capture hashes. Redundant variants and large arrays are not copied. This is source/LP/packet and negative allocator evidence only; it is not a passing mass allocation, native run, or physical/human qualification.
