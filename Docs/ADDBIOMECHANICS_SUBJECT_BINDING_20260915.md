# AddBiomechanics subject binding

The acquisition receipt now has a second boundary: it is joined to the
current Human evidence graph and the compiled MyoSim rigid-body mass owner by
`numi human`'s subject-binding command. This step prevents a source reference
subject from being silently relabeled as the runtime subject.

Run it from the repository root:

```sh
.numi/commands/human-addbiomechanics-subject-binding \
  --output Docs/media/addbiomechanics-subject-binding-20260915/receipt-v1.json
```

The selected Falisse2017 subject is 43 years old, 1.78 m, and 65.5 kg. The
current compiled rigid-body owner sums to 97.13195176621342 kg across 103
bodies. The binding therefore records a 31.63195176621342 kg difference and
keeps `same_scaled_mechanical_subject=false`. The difference is a required
subject-scaling and mass/inertia reconstruction step, not a correction that
can be hidden in the receipt.

The binding confirms that four measured gait/stair reference tables are
available and that standing and recovery references are absent. It does not
create Numi predictions, fit activation or materials, assign organ/blood/
tissue/fat mass, or qualify anatomical loading, standing, recovery, or
walking. Those gates remain open until the runtime is rebuilt for this subject
and compared on disjoint measured trials.
