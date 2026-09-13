# Cardiac cavity blood-mass candidate — 2026-09-14

The source-preserving right-heart partition now has a bounded mass-budget
contract in `numilab_human.cardiac_blood_mass_candidate`. It consumes the two
exact disjoint geometric candidates, the four-cavity/body-link bridge, and the
CVSim21 aggregate blood owner. Each candidate is bound to the four hydraulic
chambers without changing CVSim parameters or selecting a biological interface.

The contract uses the retained `1060 kg/m³` engineering density candidate. Its
provenance remains `engineering_candidate_unresolved`; no subject measurement
or source calibration is claimed. The four chamber hydraulic initial volume is
`0.00037711543019560193 m³`, giving a candidate mass budget of
`0.39974235600733804 kg`. Both right-heart conventions conserve that same
hydraulic budget and expose their hydraulic-to-surface volume scale factors.

Those scale factors are comparison metadata. They do not turn a surface
integral into a physical lumen volume, and every candidate row keeps
`physical_volume_owner`, `mechanical_mass_owner`, and `tissue_exchange_owner`
null. The receipt therefore closes source identity, candidate geometry, four
hydraulic bindings, and zeroth-moment budget accounting while retaining the
required gates for physical volume, blood/tissue transfer, activation,
materials, calibration, standing, and walking.

Reproduce it with:

```sh
PYTHONPATH=src python3 -m numilab_human.cardiac_blood_mass_candidate \
  --partition Docs/media/cardiac-partition-20260912/partition.json \
  --bridge Docs/media/organ-blood-cavity-bridge-20260913/bridge.json \
  --body-links Docs/media/organ-cardiac-cavity-body-link-20260913/body-links.json \
  --cvsim-config config/cvsim21-source.v1.json \
  --blood-owner config/cvsim21-blood-mass-owner.v1.json \
  --output Build/cardiac-blood-mass-candidate/receipt.json
```

The immutable [receipt](media/cardiac-blood-mass-candidate-20260914/receipt.json)
is source-candidate evidence, not anatomical physical qualification. The
same bytes were produced by the wrapper and the dispatcher in a detached Mac
mini worktree at `f4f912122ddc44f97c518e5c46bc00832ba5ace0`; the retained
[Mac manifest](media/cardiac-blood-mass-candidate-20260914/macmini/manifest-v3.json)
records the `d6d869f52f3fcedb1d9901df9a45c125376dc06732e76a5980537e85c67f6178`
receipt hash and the [initial Python-3.9 failure](media/cardiac-blood-mass-candidate-20260914/macmini/failed-default-python.json).
The completion ledger remains partial until one interface is supported by
matched cardiac phase/body registration, source density and unloaded/loading
data, tubular wall mechanics, conservative accepted-step tissue exchange, and
held-out subject validation.

The updated [gap-execution report](media/gap-execution-20260914/report-cardiac-blood-mass-final-v1.json)
keeps the integrated qualification state `not_assessed` and retains the
candidate as evidence for the systemic-physiology workstream only.
