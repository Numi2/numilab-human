# Muscle route mass partition candidate — 2026-09-15

The new route-mass partition compiler joins the source 416-route identity table
to the 60 single-closed muscle surface mass candidates. Each closed surface is
split equally across its explicit route incidences, producing 82 route
allocations across 78 routes while retaining the 238 routes without a closed
surface and the 88 unadmitted surfaces.

The resulting candidate closes the source budget at `6.859582804936879 kg`
and `0.006471304532959317 m3` with no double counting. The allocation is an
engineering bookkeeping policy, not measured fibre anatomy. It does not create
a physical volume or mechanical mass owner, transfer active force, calibrate
activation/materials, or add fat geometry.

The immutable receipt is
[`receipt-v1.json`](media/muscle-route-mass-partition-candidate-20260915/receipt-v1.json).
Reproduce it with:

```sh
PYTHONPATH=src python3 -m numilab_human.muscle_route_mass_partition_candidate \
  --output Docs/media/muscle-route-mass-partition-candidate-20260915/receipt-v1.json
```
