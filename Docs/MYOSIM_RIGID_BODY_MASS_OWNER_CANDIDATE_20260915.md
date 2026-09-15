# MyoSim rigid-body mass owner candidate

This candidate records the compiled rigid-body mass authority for the pinned
`myofullbody` model. It is generated from the source environment’s MuJoCo
export and preserves the body IDs, parent tree, masses, inertias, and source
revision in a compact, immutable manifest. It gives the cross-domain Human
ledger one owner for the rigid-body mass rows, so organ, blood, fat, skin,
tendon, and skeletal-muscle tissue candidates cannot be silently added to the
same rigid-body budget.

The source export was generated with:

```sh
PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m numilab_human.myosim_export \
  --sources Sources --output /private/tmp/myofullbody-export-20260915.json
```

The resulting export is source-bound to MyoSim revision
`33c89c2bde282553dde3f526768eb3bdcfaa7649`, archive SHA-256
`280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975`, and
MuJoCo `3.12.0`. Its SHA-256 is
`35e90b5625efbb8ad9cab936c29238c0e1760a3821e52e9ffbc96198879b45af`.

The checked-in source manifest retains the fields required to revalidate the
compiled ledger:

```sh
PYTHONPATH=src .venv-mujoco312/bin/python \
  -m numilab_human.myosim_mass_owner_candidate \
  --source-manifest Docs/media/myosim-mass-owner-20260915/source-manifest-v1.json \
  --output Docs/media/myosim-mass-owner-20260915/receipt-v1.json
```

The receipt binds 103 non-world bodies, 96 mass-bearing bodies, seven zero-mass
bodies, and a compiled mass sum of 97.13195176621342 kg (the source export’s
floating-point sum of the expected 97.13195176621338 kg). It also binds the
128 velocity coordinates, 129 position coordinates, 416 muscle actuators, and
the 103-row parent tree.

This is a rigid-body source ledger. It does not claim anatomical organ or
vascular lumen volume, blood mechanical mass, fat or skin mass, skeletal-muscle
tissue volume, compliant soft-tissue mechanics, material calibration, subject
scaling, standing, recovery, or walking. The body-composition integration keeps
those ownership gates false while recording this source owner separately.

## Qualification result

The candidate is `partial`: compiled mass, inertia, body-tree identity, and one
rigid-body owner per compiled body pass source-bound checks. A whole-body
dynamic mass-matrix qualification still needs the monolithic articulated solve,
anatomical support loading, muscle/tendon equilibrium, material data, and
behavioral validation described in the completion ledger.
