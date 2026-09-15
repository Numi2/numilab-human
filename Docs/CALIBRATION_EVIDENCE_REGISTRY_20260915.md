# Calibration and unresolved-material evidence registry

The registry joins the evidence needed for the remaining one-male Human
calibration work without treating a candidate as a production owner. Run it
with:

```sh
PYTHONPATH=src python3 -m numilab_human.calibration_evidence_registry \
  --output Docs/media/calibration-evidence-registry-20260915/receipt-v2.json
```

The eight rows cover the conditional finite-hold cartilage fit, source-bound
muscle recruitment, measured AddBiomechanics references, organ tissue mass,
zeroth-moment blood/tissue transfer, skeletal-muscle tissue mass, explicit
adipose-source absence, and the native activation diagnostic. Each row retains
its source hash, scope, fit flag, held-out flag, and production-owner flag.

The cartilage row has six training points and three same-plug held-out points,
but its preloaded reference, density, relaxation and native boundary-value
response remain unresolved. The recruitment row is an offline force-balance
candidate, not measured activation calibration. The AddBiomechanics row is a
real 43-year-old male reference with gait and stair tables, but no Numi
prediction comparison, and the current 97.13195176621342 kg runtime owner does
not match the 65.5 kg reference subject.

Organ, blood, muscle and fat rows preserve the existing boundaries: candidate
mass is not admitted to mechanics; blood transfer has no anatomical lumen or
capillary owner; 88 of 148 muscle surfaces remain unadmitted; and the source
package has no adipose geometry or mass input. The native activation row keeps
root/body-weight closure separate from the open articulated residual.

The receipt is therefore `partial` by construction. It closes evidence
provenance and dependency ownership, while keeping activation calibration,
material calibration, subject calibration, anatomical blood-mass transfer,
standing, recovery, walking, and integrated qualification false.
