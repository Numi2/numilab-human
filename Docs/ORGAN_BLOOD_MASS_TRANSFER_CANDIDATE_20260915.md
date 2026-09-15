# Regional blood/tissue mass-transfer candidate

The new `organ-blood-mass-transfer-candidate` command binds the seven source
organ beds to the existing CVSim21 arterial/transit/venous owners and to the
admitted regional tissue mass candidates. On the exact 12.5 microsecond clock,
it advects blood mass through each bed and applies a signed bounded exchange
between transit blood and the tissue candidate. The two phase schedule drives
both blood-to-tissue and tissue-to-blood transfers; the candidate restores a
rejected step before any state or trace is published.

The requalification receipt is
[`receipt-v3.json`](media/organ-blood-mass-transfer-20260915/receipt-v3.json).
The 512-attempt run accepts 511 steps and rejects step 37. Mass and owned
volume residuals remain inside the source candidate tolerances, both transfer
directions are observed, and the accepted state trace is deterministic.
The same compiled candidate was stepped on the physical Mac mini M4 Pro with
Python 3.9.6; [`macmini-step.json`](media/organ-blood-mass-transfer-20260915/macmini-step.json)
records the matching 511/1 transaction and conservation result. That is a
platform stepping check only and carries no native Matter qualification.

This is a zeroth-moment transport and ownership subgate. It does not create an
anatomical vessel lumen or capillary network, promote blood or tissue to the
rigid-body mass matrix, calibrate density or exchange coefficients, or qualify
organ mechanics, physiology, standing, recovery, or walking. The phase and
fraction parameters are explicitly unresolved engineering candidates.

Reproduce it with:

```sh
PYTHONPATH=src python3 -m numilab_human.organ_blood_mass_transfer_candidate \
  --reject-step 37 \
  --output Docs/media/organ-blood-mass-transfer-20260915/receipt-v3.json
```
