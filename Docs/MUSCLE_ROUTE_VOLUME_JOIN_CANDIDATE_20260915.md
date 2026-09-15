# Muscle route-to-volume incidence candidate — 2026-09-15

`muscle-route-volume-join-candidate.1` joins the complete 416-actuator source
route table to the NHTISS4 surface inventory and the 60 single-closed muscle
surface-volume candidates. It binds 78 routes to 82 closed-geometry
incidences, retains the existing 238 routes without any surface binding, and
keeps the 148 muscle plus two tendon surface identities visible.

The immutable receipt is
[`Docs/media/muscle-route-volume-join-candidate-20260915/receipt-v1.json`](media/muscle-route-volume-join-candidate-20260915/receipt-v1.json).
Each volume row carries the source member hash and matched route IDs. A
surface matched to multiple routes remains shared incidence; no volume is
partitioned between routes.

This is a geometry/identity hand-off. It does not assign density, skeletal
muscle mass, constitutive material, active volumetric force, activation
calibration, fat or skin geometry, or standing/walking authority.

Validation:

- `tests/test_muscle_route_volume_join_candidate.py`: 3 tests pass.
- The `human muscle-route-volume-join-candidate` CLI writes an immutable
  canonical receipt and rejects route-owner promotion or source-hash drift.
