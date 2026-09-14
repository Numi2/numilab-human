# Regional tissue oxygen exchange candidate, 14 September 2026

This increment joins the regional blood transport candidate to a conserved
oxygen amount for the same seven source organ beds: right/left lung,
right/left kidney, stomach, pancreas, and liver. Each bed has arterial,
transit, venous, and tissue amount state. Oxygen is advected with the pinned
CVSim21 source flow, then a bounded two-way exchange is applied between the
transit amount and the admitted organ surface-volume candidate.

The run uses the canonical `12.5 µs` clock for 512 attempted steps, rejects
attempt 37, accepts 511 steps, and performs 10,731 amount transfers: two
blood advections and one tissue exchange for each accepted bed step. Blood
mass, hydraulic volume, and oxygen amount conserve within the receipt
tolerances, and the rejected candidate leaves the accepted state unchanged.
The immutable receipt is
[`receipt-v1.json`](media/organ-tissue-exchange-candidate-20260914/receipt-v1.json),
SHA-256 `6ad5993f2147a14643e8b0d234b7a873f060ddad444029d3c21266bf8497e58d`.

The initial concentrations, partition coefficient, clearance fraction, and
tissue volume are explicitly `engineering_candidate_unresolved` values. The
surface-volume candidates remain without a physical tissue-volume owner, and
the exchange has no capillary lumen, vessel wall, gas law, metabolic reaction,
oxygen consumption, organ mechanics, calibrated material, or subject
parameter. The candidate therefore closes a source-bound amount/conservation
subgate while remaining partial physiology evidence.

The refreshed [gap execution report](media/gap-execution-20260914/report-organ-tissue-exchange-candidate-current-20260914.json)
retains `integrated_qualification=not_assessed`.

Reproduce it with:

```sh
./.numi/commands/human organ-tissue-exchange-candidate \
  --steps 512 --reject-step 37 \
  --output Docs/media/organ-tissue-exchange-candidate-20260914/receipt-v1.json
```
