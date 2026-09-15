# Skeletal muscle tissue mass candidate — 2026-09-15

This candidate applies an explicit `1060 kg/m3` engineering density to the 60
single-closed source muscle volumes. It produces a reproducible `6.859582804936875
kg` candidate budget while retaining the density as unresolved and leaving the
mechanical owner count at zero.

The 88 remaining muscle surfaces (six closed multi-component and 82 topology
defective) stay unadmitted. The result is not a disjoint partition, FEM mass,
MyoSim rigid-body replacement, active-force transfer, activation calibration,
fat geometry, or behavior qualification.

Reproduce it with:

```sh
PYTHONPATH=src python3 -m numilab_human.muscle_tissue_mass_candidate \
  --output Docs/media/muscle-tissue-mass-candidate-20260915/receipt-v1.json
```
