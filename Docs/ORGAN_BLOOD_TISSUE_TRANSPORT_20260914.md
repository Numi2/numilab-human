# Regional organ blood transport candidate, 14 September 2026

This increment binds seven source organ regions to the pinned CVSim21 blood
owner: right/left lung, right/left kidney, stomach, pancreas, and liver. The
regional share for each paired or grouped bed is proportional to its admitted
organ surface-volume candidate. Each bed routes source arterial blood through
an explicit engineering transit volume and into the matching source venous
owner. The source owner is partitioned once; it is not duplicated for each
organ.

The bounded run uses the canonical `12.5 µs` clock for 512 attempted steps,
rejects attempt 37, accepts 511 steps, performs 7,154 arterial-to-transit and
transit-to-venous transfers, and conserves blood mass and volume within the
receipt tolerances. The immutable receipt is
[`receipt-v1.json`](media/organ-blood-tissue-transport-20260914/receipt-v1.json),
SHA-256 `5b04d8df64aaa83f4b5fd623b4e4ee730fc543783c02f90cc0aab508ae0c7e32`.

The transit time is an explicit `0.25 s` engineering candidate with unresolved
provenance. It is not a capillary measurement. Source organ surface moments
remain candidate geometry, and the source CVSim21 density remains the
unresolved `1060 kg/m³` engineering value.

This closes a source-bound zeroth-moment regional transport and rollback
subgate. It does not qualify anatomical vessel tubes or lumens, first/second
blood spatial moments, tissue oxygen/metabolite exchange, organ mechanics,
mechanical mass, material calibration, subject physiology, standing, or
walking. The new `blood.spatial_transport` registry task therefore remains
partial until those owners are supplied.

The refreshed [gap execution report](media/gap-execution-20260914/report-organ-blood-tissue-transport-current-20260914.json)
has 48 registry tasks and remains `integrated_qualification=not_assessed`.

Reproduce it with:

```sh
./.numi/commands/human organ-blood-tissue-transport \
  --steps 512 --reject-step 37 \
  --output Docs/media/organ-blood-tissue-transport-20260914/receipt-v1.json
```
