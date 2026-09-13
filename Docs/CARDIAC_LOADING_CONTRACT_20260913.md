# Cardiac material, activation, support, and loading contract, 13 September 2026

The contract receipt at
`Docs/media/cardiac-loading-contract-20260913/contract.json` pins the complete
Rodero case18 source configuration and preserves the supplied passive material,
activation, and loading values without turning them into a native anatomical
wall owner.  Its source configuration hash is
`23c931fef53edced85e0e0a36c73d8490dddb87db6bd988482a2cdfc5a1442cc`; the
contract identity is
`7b502575911dd1d5267959cddb1e1aa47fa570e148137d813a7be2bc56a38ab2`.

The source activation parameters are an activation-based Tanh Stress model
with a 20 ms electromechanical delay, 120 kPa peak isometric tension, 50 ms
contraction and relaxation time constants, and a 550 ms transient.  The
loading record retains LV/RV endocardial pressures of 1.6/0.8 kPa, aortic and
pulmonary arterial pressures of 77/17.4 mmHg, constant atrial preload, and the
source systemic/pulmonary three-element Windkessel values.  Zero forward
aortic and pulmonary valve resistances remain exact source values; no epsilon
resistance is substituted.

The compiler emits six fail-closed admission gates:

| Gate | Source status | Admitted |
| --- | --- | --- |
| stress-free reference configuration | absent from case18 archive | no |
| inertial density | absent from retained article/S4 | no |
| epicardial Robin coefficients and spatial field | described but not quantified | no |
| venous anchor interpretation | article and S4 descriptions conflict | no |
| source reference trajectory | CARP inputs/time-resolved trajectory absent | no |
| zero-forward valve native admission | current positive resistive-edge owner cannot represent it | no |

Closure labels 11--24 remain explicitly unresolved source closures/borders.
The source material classes, hand-tuned active values, and shared loading
parameters are source specifications, not subject-specific calibration.  The
receipt therefore records `native_anatomical_wall_admitted=false`,
`closure_materials_resolved=false`, `stress_free_reference_supplied=false`,
`source_activation_time_field_supplied=false`, and
`subject_specific_calibration=false`.

Use `numi human-cardiac-loading-contract --output <new-receipt.json>` to
recompute the immutable contract.  The independent verifier reports six
unresolved gates and fourteen retained closure labels.  This increment closes
source parameter bookkeeping for supports/loading, activation, and materials;
it does not close anatomical registration, unloaded reconstruction, native
wall stepping, blood/tissue coupling, or standing/walking.
