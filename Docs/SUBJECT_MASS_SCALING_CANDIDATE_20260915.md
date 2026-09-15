# Subject mass scaling candidate

The acquired Falisse2017 subject weighs 65.5 kg while the current compiled
MyoSim rigid-body owner weighs 97.13195176621342 kg. The subject-binding gate
keeps that mismatch open. This increment emits a deterministic per-body
mass-only normalization with factor `0.6743404081661175`, preserving the 103
body tree and closing the target total at 65.5 kg.

Generate the candidate with:

```sh
.numi/commands/human-subject-mass-scaling \
  --output Docs/media/subject-mass-scaling-20260915/receipt-v1.json
```

The receipt includes source and scaled mass/inertia rows, the original owner
identities, and the scalar closure error. It intentionally uses
`uniform_mass_only_fixed_geometry`: the geometry factor is one and inertia is
scaled with mass. No measured segment composition, subject-specific geometry,
inertial validation, material fit, organ/blood/tissue/fat/muscle ownership, or
native runtime admission is inferred. The candidate is the explicit input for
the next native subject-scaled mass/inertia solve, not its qualification.
