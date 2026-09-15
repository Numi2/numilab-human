# One-male body-composition mass ledger — 15 September 2026

The mass ledger joins the existing one-male source records into one explicit
ownership boundary. The bound subject is Falisse2017 `subject_1`: male, age 43,
height 1.78 m, measured mass 65.5 kg. The compiled MyoSim source owner is
97.13195176621342 kg across 103 bodies; the uniform mass-only handoff closes
the scalar target at 65.49999999999994 kg with a `-5.684341886080802e-14 kg`
closure error.

The ledger also records the current candidate scopes: 2.7781575033215167 kg of
organ surface candidates, 5.079334472359306 kg in the seven-bed blood/tissue
transfer state, 0.39974235600733804 kg in the cardiac hydraulic candidate,
0.25057866444098925 kg in the six-vessel surface-moment candidate,
0.11369939548001184 kg in the regional costal tissue partition, and
6.859582804936875 kg / 0.006471304532959316 m³ in the skeletal-muscle route
partition. The fat source receipt contains zero fat surfaces, volume
candidates, or mass candidates.

Those values are deliberately not summed. Their source identities and scopes
overlap, several use unresolved engineering density, and none is a production
physical owner. The receipt therefore emits
`candidate_mass_sum_status=forbidden_until_interdomain_partition` and
`candidate_mass_admitted_to_dynamics=false`. This closes the bookkeeping
question while keeping organ mechanics, anatomical blood transfer, soft-tissue
mass, fat geometry, activation/force transfer, materials, segment/inertia
calibration, standing, recovery, and walking unqualified.

The immutable receipt is
[`Docs/media/body-composition-mass-ledger-20260915/receipt-v1.json`](media/body-composition-mass-ledger-20260915/receipt-v1.json).
Generate it with:

```sh
PYTHONPATH=src python3 -m numilab_human.body_composition_mass_ledger \
  --output Docs/media/body-composition-mass-ledger-20260915/receipt-v1.json
```

The current Human evidence join binds this ledger in
[`Docs/media/current-human-evidence-join-20260915/receipt-v5.json`](media/current-human-evidence-join-20260915/receipt-v5.json).
