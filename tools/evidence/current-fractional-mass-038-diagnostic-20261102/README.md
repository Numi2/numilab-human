# Current fractional mass mapping: 038 diagnostic and source-only controls

Status date: 2026-11-02. This package records the current 030d source-control route, its failed full-graph run, and the source-only postpass compatibility code. It contains no mesh, NPZ, compiled binary, or full capacity-witness payload.

## Actual run result

The canonical 038 run `out-006` passed launch preflight, then exited 6 with `INFEASIBLE_FULL_GRAPH_CUT`. The deterministic full minimum-flow phase exhausted residual reachability. It sent 12.968739234433393 kg against 12.998725017533202 kg of minimum child demand; the cut reports 0.029985783099903126 kg unmet. This is a failed allocator diagnostic, not a timeout or a physical admission. The run records `native_steps=0` and `physical_admission=false`. No allocation/pass receipt was produced.

The only full-cut payload is retained at its original absolute runtime path and is hash-referenced in `evidence/038/out-006/full-cut-manifest.json`; it is not copied here because it is 378,689,260 bytes compressed (1,498,523,142 bytes uncompressed). Its compact cut header and both manifests are included. The v15 support report is an optimistic capacity upper bound, not a proof of feasible mass transport.

## Source and tests

The versioned 038 builder, run-manifest generator, launch wrapper, 030d contract reader, and their unit tests are in `source/038`. The 030d CLI source and exact source-control manifest are included, but the executable and dependency headers remain at the original pinned paths recorded by the manifests.

The 015 adapter reuses the frozen 014 deterministic replay over the recorded output CSVs; it does not execute the allocator. It handles 038's two run-control hashes as separate authenticated refs, preserving 006's exact 27-key app environment. It must reject the failed `out-006` run, and no 015 receipt was generated from that attempt. If a later run passes, the source-only projection remains separate from the original `projection_report` input; a finalizer may bind it as `source_projection_015` after generation.

Focused results:

- 038 source-control/launcher tests: 13 passed.
- 015 compatibility tests: 12 passed.
- 010i v2 identity/run-control tests: 8 passed.
- Python compile, Ruff F821/F823, and the 010i shell syntax check passed.

The first 010i test invocation from the compact package failed because the adjacent pinned 009 contract module had not yet been copied. The package was completed with the exact pinned 009 contract and launch-003 dependency; the final 010i suite then passed 8/8. `checks/010i-v2-first-package-closure-failure.json` preserves that packaging diagnostic.

The 010i v2 identity config included here is pre-mass only; its `mass_admission` remains null. It is not native or physical qualification.

## Provenance and limits

`artifact-manifest.json` records the copied bytes and hashes. The copied run manifest binds the exact 030d executable, source control, closure, inputs, and output directory. Large source inputs and the v15 witness CSVs remain at their original paths and are identified by hash and size in the manifest or run config; they are not duplicated into this repository.

This package does not claim successful mass feasibility, full recorded-flow qualification, native execution, physical admission, or GPU motion. An `out-007` iteration is not launched or represented here; preparation waits for the current spare-point support route to pass a feasibility screen against the retained `out-006` cut.

## Packaging check note

`git diff --cached --check` reports only a final blank line in six byte-pinned upstream source copies: `source/010i-v2/prepare_current_native_1025_direct_o_011.py`, `source/030d/NumiHumanCurrentFractionalMassMappingCLI_030d.cpp`, `source/038/recorded_flow_contract_010_030d.py`, `source/038/run_canonical_038.py`, `source/038/test_run_canonical_038.py`, and `source/038/test_source_control_038.py`. Those lines are preserved to keep each source SHA identical to its frozen owner; no other whitespace findings were reported.
