# Explicit blood-mass transfer owner — 2026-09-13

The Human physiology fixture now has a separate mass-owner step in
`numilab_human.blood_mass_step`. It binds every compiled blood compartment and
tissue reservoir to one physical-volume owner, requires explicit fixture
density and initial mass, advects mass through hydraulic flow, applies a
declared bidirectional blood/tissue exchange schedule, and rolls back hydraulic
and mass state together when a candidate is rejected.

The retained [receipt](media/blood-mass-transfer-step-20260913/receipt.json)
completed 32 attempted steps at `100 µs`, accepted 31 after one deliberate
rollback, and observed both blood-to-tissue and tissue-to-blood transfer. Total
owned mass was `0.00418 kg` at both endpoints with zero reported mass residual;
owned volume drift was `-8.47e-22 m³`.

Reproduce it with:

```sh
PYTHONPATH=src python -m numilab_human.cli blood-mass-step \
  --graph config/physiology-passive-fixture.v1.json \
  --owners config/blood-mass-transfer-fixture.v1.json \
  --sources Sources --steps 32 --timestep-seconds 0.0001 \
  --reject-step 9 --output /absolute/path/receipt.json
```

This is a fixture-only ownership and conservation subgate. The densities,
compartment geometry, exchange schedule, and tissue reservoir are manufactured
numerical inputs. The owner does not admit anatomical vessel tubes or lumens,
subject blood density, organ mechanics, perfusion, cardiac activation,
material calibration, subject calibration, standing, or walking.
