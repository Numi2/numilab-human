# Current Human evidence join — 2026-09-15

The current source-bound join is recorded in
[`Docs/media/current-human-evidence-join-20260915/receipt-v1.json`](media/current-human-evidence-join-20260915/receipt-v1.json)
and is reproducible with:

```sh
PYTHONPATH=src python3 -m numilab_human.current_human_evidence_join \
  --output Docs/media/current-human-evidence-join-20260915/receipt-v1.json
```

The receipt binds the pose-24 native force audit, the complete 128-row force
ledger, the matching 512-step passive release, the current source-composition
candidate, and a regional blood/oxygen transaction requalified on the same
native owner. The physical owner is the Mac mini M4 Pro, branch
`numi-human-passive-stand-20260915`, commit
`7625ec565e086faf0dcd349846dadc2d22d65e86`, with mechanics binary hash
`e78e6efd31471eba0839ebf75674d64395e1e804a9751be7f52c1fb420113fe5`.

The static mechanics subgates now have one joined evidence record: 157 bodies,
128 generalized rows, six published source contributions per row, maximum
absolute force residual `2.2805963908467675e-6 N`, maximum assembly error
`6.773910859320109e-14`, maximum closure ratio `1.5606352982019566e-5`, and
six active source support witnesses. The passive release completes 512 exact
clock steps with `0 m` penetration and bitwise replay; its peak acceleration is
`4.85533761978 m/s2`. These are bounded release and static closure results,
not sustained standing.

The regional transaction is also rebuilt on commit `7625ec56`: 21 source
compartments, 24 connections, seven beds, 512 attempted steps, 511 accepted
steps in environment zero, 37 rejected steps, exact `12,500 ns` clock,
bitwise rollback/replay, and maximum relative volume, blood-mass and oxygen
residuals of `5.711629397e-7`, `5.711629397e-7`, and `1.057184875e-6`.
The candidate density remains `1060 kg/m3`; this is amount transport and
conservation evidence, not an anatomical lumen or mechanical blood-mass owner.

The remaining gates stay explicit: force/state refinement, anatomical
supports/loading, dynamic foot contact, measured activation calibration,
anatomical blood mass transfer, organ mechanics, fat and skeletal-muscle
physical ownership, calibrated materials, subject calibration, sustained
standing, perturbation recovery, and walking. Candidate masses remain barred
from the rigid-body dynamics.

## v2 evidence join

The v2 receipt additionally hash-binds the v13 body-composition integration,
the physical-M4 dynamic force-component audit, and the exact-clock regional
blood/tissue mass-transfer candidate. The dynamic audit exposes all 128 initial
components and retains a 64-step, `0.115904301405 m/s2` release with
`0.03216604835060366 N` maximum initial residual, zero penetration, six active
source contacts, and bitwise replay; it is diagnostic evidence and does not
qualify temporal force convergence. The mass-transfer candidate accepts 511 of
512 attempts, rejects step 37 atomically, conserves mass and volume, and
exercises both directions. Anatomical lumen/capillary geometry, mechanical
blood/tissue ownership, calibrated density, materials, subject calibration,
sustained standing, recovery, and walking remain open.

Reproduce the current receipt with:

```sh
PYTHONPATH=src python3 -m numilab_human.current_human_evidence_join \
  --output Docs/media/current-human-evidence-join-20260915/receipt-v2.json
```

## v3 evidence join

The v3 receipt uses the v14 body-composition integration. In addition to the
existing mechanics and exact-clock blood/tissue candidates, it therefore
hash-binds the compiled MyoSim rigid-body mass ledger: 103 non-world bodies,
96 mass-bearing rows, seven zero-mass rows, and `97.13195176621342 kg` of
compiled rigid-body mass. The source revision and export hash are retained;
anatomical organ, blood, fat, skin, soft-tissue material, whole-body dynamic
mass-matrix, subject-calibration, standing, recovery, and walking gates remain
false.

Reproduce the current receipt with:

```sh
PYTHONPATH=src python3 -m numilab_human.current_human_evidence_join \
  --output Docs/media/current-human-evidence-join-20260915/receipt-v3.json
```

## v4 evidence join

The v4 receipt additionally binds the isolated coupled-velocity diagnostic and
its canonical 128-DoF dynamic force ledger. The experimental M4 Pro replay
completes 64 exact-clock steps with six active contacts, zero penetration,
bitwise endpoint replay, `0.115922890604 m/s2` peak acceleration, and
`0.032620927143 N` dynamic residual. The ledger reconstructs all six force
owners to `8.53e-14 N` assembly error, while its low-load internal normalized
residual reaches `0.0161364514`; these records remain diagnostics and do not
promote force convergence, standing, recovery, walking, anatomical loading,
activation calibration, blood mechanical ownership, materials, or subject
calibration.

Reproduce the current receipt with:

```sh
PYTHONPATH=src python3 -m numilab_human.current_human_evidence_join \
  --output Docs/media/current-human-evidence-join-20260915/receipt-v4.json
```

## v5 evidence join

The v5 receipt additionally binds the one-male body-composition mass ledger.
It identifies Falisse2017 `subject_1` as a 43-year-old male at 1.78 m and
65.5 kg, closes the scalar mass-only handoff with a
`-5.684341886080802e-14 kg` residual, and records the 97.13195176621342 kg
compiled source mass and the 31.63195176621342 kg pre-scaling mismatch.
Organ surface, blood/tissue, cardiac, vessel, regional tissue, skeletal-muscle
and fat quantities are retained as separate candidate scopes; their mass sum
is forbidden until a disjoint interdomain partition exists, and no candidate
mass is admitted to rigid-body dynamics. The receipt therefore advances
one-male mass bookkeeping while leaving physical soft-tissue ownership,
materials, activation transfer, anatomical blood mechanics, standing,
recovery, walking, and integrated qualification false.

Reproduce the current receipt with:

```sh
PYTHONPATH=src python3 -m numilab_human.current_human_evidence_join \
  --output Docs/media/current-human-evidence-join-20260915/receipt-v5.json
```
