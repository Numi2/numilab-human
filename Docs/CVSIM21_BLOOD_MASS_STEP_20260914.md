# CVSim21 aggregate blood-mass step — 2026-09-14

The pinned CVSim21 source graph now has a source-bound mass owner at
`numilab_human.cvsim21_blood_mass_step`. It binds all 21 source aggregate
compartments to their existing `source_blood:CVSim21:v*` volume owners, uses the
retained 5,150 mL source budget, advects mass with the 24 source hydraulic
connections, and restores hydraulic and mass state together when a candidate is
rejected.

The owner uses an explicit `1060 kg/m³` engineering candidate. CVSim21 does not
provide a blood-density measurement, so the owner records that value as
`engineering_candidate_unresolved` and keeps anatomical registration, tissue
exchange, mechanical mass/inertia, material calibration, and subject calibration
false. The step is therefore a source aggregate mass-transfer subgate; it does
not turn regional hydraulic labels into organ or vessel lumen mechanics.

The exact-clock receipt attempted 512 steps at 12,500 ns, deliberately rejected
attempt 37, and accepted 511 steps. It conserved the 5.459 kg candidate mass
within `1.78e-15 kg` and the 5.150 L source volume within `3.47e-18 m³`.
The shorter receipt attempted 32 steps at 100 µs and rejected attempt 9.

Reproduce the exact-clock run with:

```sh
PYTHONPATH=src python3 -m numilab_human.cvsim21_blood_mass_step \
  --config config/cvsim21-source.v1.json \
  --owners config/cvsim21-blood-mass-owner.v1.json \
  --steps 512 --timestep-seconds 0.0000125 \
  --require-clock-nanoseconds 12500 --reject-step 37 \
  --output Build/cvsim21-blood-mass-step/receipt-exact-clock.json
```

The retained [exact-clock receipt](media/cvsim21-blood-mass-step-20260914/receipt-exact-clock.json)
and [100 µs receipt](media/cvsim21-blood-mass-step-20260914/receipt.json) are
immutable. The focused local source, cardiac, physiology, blood-mass, gap, and
force suites pass 81 tests. Mac mini evidence is retained beside the receipts
after isolated-worktree validation.

This increment closes source aggregate zeroth-moment mass ownership and
accepted-step mass rollback only. The completion ledger remains partial: source
anatomical registration, disjoint lumen/tissue partition, density/material
identification, perfusion and tissue exchange, activation calibration, force
convergence, sustained standing/recovery/walking, and held-out physiology still
require their own source data and validation.
