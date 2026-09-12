# Four-chamber cardiac source reproduction — 12 September 2026

The complete curated Shi–Hose hydraulic circulation now runs on physical Apple M4 Pro Metal inside the existing Matter transaction: four driven chambers, four one-way valves, and systemic and pulmonary vascular storage and flow. Native revision `f5ce0ce563d54314f7226bb56b8f323de8dd69c3` extends the passive owner at `1594f7aff5503aab96ab9de6e77b6d4f8fa4f1cd`. This closes the bounded source-model implementation and numerical reproduction gate. Absolute vascular blood volume, physiological calibration, anatomical organ mechanics, standing and walking remain unqualified.

## Source and ownership

The [curated Physiome exposure](https://models.physiomeproject.org/exposure/c49d416ae3a5132882e6ea7479ba50f5), attributed to Yubing Shi, Rod Hose and the repository contributors, is pinned to revision `a679cdc2e97429fb5280af8132c119758626c1f2`. Its archive SHA-256 is `91d6b586c0caaa0fd59ec21cef873f1348563af0033210fbadfe84921a52782d`. All 15 imported CellML files, original metadata and CC BY 3.0 attribution are retained under [third_party](../third_party/physiome/shi_hose_2009/README.md).

The [Human author](SHI_HOSE_SOURCE_AUTHORING_V2.md) resolves 59 source parameters and their units from the pinned XML and emits `HumanPack.physiology-native.v2`. It retains the source pressure conversion of exactly 133 Pa, angular literal `3.14159`, 500 mL initial ventricular volumes, and initially negative venous return. The six vascular pressure states map to signed compliance storage `C·P`; the source does not supply their unstressed or absolute blood volumes. They therefore cannot supply species dilution volumes, tissue exchange or whole-body blood mass. The existing 18-region anatomical template retains its 276 unresolved values.

Matter owns all persistent state and physical stepping. Human adds authoring and evidence tools. An independent offline FP64 C++ Dormand–Prince reference evaluates equations generated directly from the source CellML imports, aliases and MathML; it does not reuse the native hydraulic residual or Human parameter lowering. Python translates source, launches executables and compares retained outputs; it does not integrate physical state.

The curated source uses instantaneous square-root valves. It does not reproduce the original papers' valve leaflet dynamics. Zero-storage series resistances remain explicit in the independent source reference and are eliminated algebraically in the native circuit.

## Native implementation and repaired failure

Periodic ventricular and atrial elastance, passive resistance/inertance and valve equations occupy the existing Newton/FGMRES system. A per-environment unsigned 128-bit binary clock advances once per accepted physical microtick. Chamber phase stays fixed during the nonlinear solve. Clock carry, overflow rejection, reset, restore, rollback and prepared-state proofs are tested; private working masks and derived elastance add no independent physical authority.

The first ten-cycle attempt failed at step 984, around 1.968 seconds. A closing valve's feasibility limiter collapsed the shared step length while its flow remained above the original residual tolerance. The [failed log](media/cardiac-source-20260912/retained-failure-first-ten-cycles-2ms.log), partial trace and diagnostic state are retained. A single-step replay reproduced the failure at 12 Newton / 64 Krylov iterations. Raising the Newton budget to 24 was diagnostic only.

The repair introduces a primal working set for directions crossing zero valve flow. A constrained valve receives an affine correction to zero while conservation and species rows consistently solve the remaining free variables. A new constraint gets a free-variable solve before pressure-based release. Bounded Armijo backtracking uses the original vascular residual when valves open. The original complementarity residual and authored tolerances still certify every accepted state. The same shared step length now also applies to rigid/support corrections when there are no continuum objects. No accepted state is repaired by clipping.

The final single-step replay and all full runs pass at the original 12/64 budget. Independent opening and closing transport fixtures check the quadratic valve solution, species transfer, zero closed-valve leakage and conservation. Transaction checks cover shared step lengths, discarded corrections, newly constrained rows and pressure-driven release.

Compatibility is **Matter ABI 27, package 12, snapshot archive 6 and accepted-state proof manifest 6**. Earlier Matter packages require recooking. Historical receipts remain tied to their original revisions.

## Measured evidence

Every native run uses two environments with bitwise-equal accepted states, exact binary clock progression, Metal API validation and zero failed steps. Source and executable hashes were recorded before and after each run and matched the final committed owner. The independent source reference integrates at the actual cooked Float32 timestep.

| Native interval | Steps | Maximum chamber volume error (m³) | Maximum vascular storage error (m³) | Maximum flow error (m³/s) | Scaled state RMS error |
| --- | ---: | ---: | ---: | ---: | ---: |
| One cycle, 2 ms | 500 | 1.724384e-6 | 1.694983e-6 | 5.813237e-4 | 0.132804 |
| One cycle, 1 ms | 1,000 | 9.784753e-7 | 9.646154e-7 | 3.325871e-4 | 0.073574 |
| One cycle, 0.5 ms | 2,000 | 5.204598e-7 | 5.132451e-7 | 1.769189e-4 | 0.039345 |
| Ten cycles, 2 ms | 5,000 | 1.724384e-6 | 1.694983e-6 | 5.813237e-4 | 0.060869 |

Equal-duration refinement spans exactly `1.0000000474974513` seconds; the ten-cycle run spans `10.000000474974513` seconds. All four error families decrease by at least 1.5× at each refinement. The ten-cycle maximum relative total hydraulic storage drift is `7.352916e-7`; its native wall time was 194.303 seconds. This small hydraulic workload does not establish whole-Human performance. Flow maxima include the source initial transient and are reported without claiming physiological accuracy.

Numerical regression ceilings were fixed before the repaired cohort, informed by retained initial-transient results. They are implementation regression envelopes, not experimental or physiological acceptance criteria. The verifier independently recomputes every metric from full traces, including individual chamber errors and vascular pressure-equivalent errors.

The independent C++ reference retains 20 cycles at tolerances `1e-10`, `1e-12` and `1e-13`, with 20,001 output samples each. Between the two tighter runs, maximum pairwise differences are below `9.83e-6 Pa`, `2.84e-13 m³` chamber volume and `5.03e-11 m³/s` flow. These are measured refinement differences, not rigorous exact-solution bounds. The final reference cycle is not declared perfectly periodic.

All **13 selected native CTests pass**, covering snapshot/archive authority, monolithic multiphysics, production rollback, stateful MPM/FEM, prepared-state proofs, compiler round trips, vascular runtime, both Human payload versions, strict input admission and cardiac transactions/source execution. Input admission covers two positive payloads and 37 negative cases while preserving outputs on rejection. Full summaries and individual test outputs are retained with source, binary and device identities.

Human validation passes **17 source-authoring tests, 23 evidence/tamper tests and 28 existing physiology tests**. The prior passive-circulation receipt still verifies at its historical revision. The actual `numi human-cardiac` dispatcher emits the byte-identical qualified source payload. The final cardiac verifier passes against all 41 retained artifacts and 38 owning Human source pins.

## Reproduce and verify

From the Human repository:

```sh
numi human-cardiac --output Build/shi-hose/closed-loop.native.json
PYTHONPATH=src python3 -m unittest discover -s tests -p test_shi_hose.py
python3 tools/verify_cardiac_source_20260912.py Docs/media/cardiac-source-20260912/receipt.json
python3 -m unittest discover -s tests -p test_cardiac_source_evidence.py
```

The native owner document, `docs/HUMAN_CARDIAC_SOURCE_V2.md` at the pinned native revision, gives compilation and physical Mac mini commands. [The independent reference README](../tools/shi_hose_reference/README.md) gives source generation and FP64 reproduction commands. The [receipt](media/cardiac-source-20260912/receipt.json) binds every retained artifact and owning source. Before-fix runs remain archival evidence and cannot satisfy final qualification roles. Tamper tests exercise source, binary, execution, clock, numerical and failure-retention admission.

## Remaining completion gates

The next organ/circulation increment needs sourced absolute vascular volumes and a nonduplicated whole-body blood mass partition, followed by anatomy-registered organ perfusion and independently calibrated observables. Active tissue exchange requires those physical dilution volumes. Deforming organ/vessel mechanics, pressure/volume/force feedback, ventilation and gas exchange, metabolism, thermal state and peripheral conduction remain separate gates. Subject calibration and held-out experimental validation remain open. Dynamic loaded standing, balance control and walking remain open in the neuromusculoskeletal workstream.
