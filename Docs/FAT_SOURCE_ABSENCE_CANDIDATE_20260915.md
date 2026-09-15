# Fat source absence candidate — 2026-09-15

The current one-adult-male Human source package does not contain an adipose
geometry or mass layer. This candidate makes that missing input explicit by
checking the sparse muscle/tendon surface receipt, the source-bound FJ2810
visual skin shell, the tissue-mass class table, and the current composition
join. All four inputs agree that fat surface count, fat volume and fat mass
are absent, and that no fat physical or mechanical owner exists.

This is an inventory and fail-closed boundary, not a fat estimate. The skin
shell is visual only, and the muscle surfaces do not define adipose tissue.
Completion requires a registered adipose source for the same male subject,
with geometry, thickness or volume, density/material observations, and a
non-overlapping mechanical ownership contract before it can enter Human
dynamics.

The immutable receipt is
[`receipt-v1.json`](media/fat-source-absence-candidate-20260915/receipt-v1.json).
Run it with:

```sh
PYTHONPATH=src .venv-mujoco312/bin/python -m numilab_human.fat_source_absence_candidate \
  --output /private/tmp/numilab-human-fat-source-absence.json
```

It remains `partial`; it does not close fat geometry/mass, material
calibration, subject calibration, standing, recovery, or walking.
