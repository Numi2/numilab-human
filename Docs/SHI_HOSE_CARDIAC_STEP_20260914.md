# Pinned Shi/Hose cardiac source step — 2026-09-14

This increment adds an executable source-model step for the exact fifteen-file
Shi/Hose CellML source already compiled by `numilab_human.shi_hose`. The step
is downstream of source compilation: it admits only the
`HumanPack.physiology-native.v2` graph with
`qualification=source_model_reproduction`, the
`closed_periodic_elastance_orifice_v2` law, ten source compartments, and ten
source connections.

The accepted candidate evaluates the source phase activation with the retained
source `3.14159` constant, computes chamber elastance and vascular compliance
pressure, evaluates one-way orifice and resistance/inertance flows, transfers
compartment volume conservatively, and advances accepted time only after the
candidate is admitted. A rejected attempt restores the full hydraulic state;
the receipt retains accepted-state and activation traces for replay.

The local receipt uses 100 attempted steps at `1e-4 s` with attempt 37
rejected. It records 99 accepted steps, one rejected step, conserved total
source compartment volume (`0.0013274 m³`, residual `0.0`), 99 open-valve
evaluations, and a deterministic accepted-state trace:

`Docs/media/shi-hose-cardiac-step-20260914/receipt.json`

The source lowering remains bound to source manifest
`3c1439ede07f5736520856f77a8cef11103e63471fa4ee264b24c8a04a90a39e`; the
receipt schema is `HumanPack.shi-hose-step-receipt.v1`. The CLI is exposed as
`numilab-human shi-hose-step` and `.numi/commands/human-shi-hose-step`.

This is a source-hydraulic reproduction subgate. It does not create absolute
vascular blood volume from storage displacement, species or dilution state,
anatomical organ or vessel mechanics, tissue exchange, material data,
subject/activation calibration, force or temporal convergence, or sustained
standing, recovery, or walking evidence.
