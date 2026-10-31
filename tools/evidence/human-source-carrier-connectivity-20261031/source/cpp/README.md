# Source-carrier LP feasibility finalizer

This is a source-only C++ adapter for a role-specific held Dinic subflow plus sparse LP positive-carrier rows. It does not invoke an optimizer and makes no optimality claim. A complete proposal is accepted only by the existing solver-free source-carrier feasibility API; this is not native execution or physical admission.

## API

Include SourceCarrierLPFeasibilityFinalizer.hpp alongside the existing application header ConservativeSpatialSourceCarrierQuadrature.hpp.

- source_carrier_lp_detail::buildAndVerify(...) combines authenticated held partial-flow rows with positive LP variables, reconstructs carrier weights and mass/measure flows, recomputes barycentric points/distances in C++, then calls canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility.
- source_carrier_lp_detail::writeExistingSourceCarrierCSVContract(...) emits the six established transport CSVs and result.json in a new/empty directory. It checks all row domains and CSR offsets before writing.
- Provenance keeps held Dinic augmentation count and LP positive-variable count separate. semanticCandidateEdgeCount is the separately pinned count from the held/LP graph records. No optimizer-replay count is synthesized.

The contract currently requires at least one held partial row and nonzero held Dinic augmentation evidence. Pure-LP proposals fail closed; zero Dinic augmentation is never fabricated.

The receipt is numi.human.source-carrier-lp-feasibility-allocation.v1 with a role-specific PASS_SOURCE_ONLY_*_SOURCE_CARRIER_FEASIBLE_ALLOCATION status, recorded_flow_feasibility=PASS_SOLVER_FREE, deterministic_optimizer_replay=NOT_RUN, physical_admission=false, and native_steps=0. The six output CSV names/headers match the existing source-carrier allocation contract. The receipt retains the legacy file-pin aliases and adds the exact current child input_region_label_sha256 used in the carrier-array digest; this is distinct from the parent label CSV pin.

## Fixture control

On SSH host numi, from this directory:

    clang++ -std=c++20 -O0 -Wall -Wextra -I /Users/numi/Work/numi-lab-723-publication-20261030/apps SourceCarrierLPFeasibilityFinalizerCheck.cpp -o SourceCarrierLPFeasibilityFinalizerCheck
    ./SourceCarrierLPFeasibilityFinalizerCheck

The fixture exercises material 15 and 16, all six CSV headers, receipt truth fields and pins, and rejection of malformed carrier maps, missing or mismatched region-label pins, attempted physical-admission metadata, wrong radius, stale old-expanded row, invalid carrier moments, cross-material support, and pure-LP/no-held-evidence input.

The independent Python verifier is verify_recorded_carrier_allocation_197_root.py in the adjacent fine-volume task directory. It retains the older deterministic-replay receipt mode and adds the explicit solver-free LP-feasibility receipt mode. It checks the composite held-plus-LP carrier packet and array digests; no full allocation was run for this control.

No real LP proposal has been finalized here. The current geometry gate blocked that run; the fixture is interface/control evidence only.
