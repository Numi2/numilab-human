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

The exact-clock companion runs 512 attempted steps at `12,500 ns` over a
`6.4 ms` attempted source horizon, rejects attempt 37, accepts 511 steps
(`6.3875 ms` accepted time), and preserves the same volume invariant. It is retained at
`Docs/media/shi-hose-cardiac-step-20260914/receipt-exact-clock.json`; the CLI
requires `--require-clock-nanoseconds 12500` for this admission.

The source lowering remains bound to source manifest
`3c1439ede07f5736520856f77a8cef11103e63471fa4ee264b24c8a04a90a39e`; the
receipt schema is `HumanPack.shi-hose-step-receipt.v1`. The CLI is exposed as
`numilab-human shi-hose-step` and `.numi/commands/human-shi-hose-step`.

The 100-step source selection was checked in an isolated Mac mini worktree at
`2a7c2b548a3b8bc406f8330b4cf190be1b8ad4ec`. The source-specific unittest
selection ran 86 tests with exit status 0, and the Mac mini CLI reproduced the
same immutable receipt SHA. The retained command/log/receipt hashes are in
`Docs/media/shi-hose-cardiac-step-20260914/macmini/manifest.json`; the dirty
shared `/Users/n/numilab-human` checkout was not changed.

The exact-clock selection was then checked at commit `f5eb3a2` in a second
isolated Mac mini worktree. Its 25 source tests passed, the 512-attempt CLI
accepted 511 steps at exactly 12,500 ns, and its receipt SHA is retained in
`Docs/media/shi-hose-cardiac-step-20260914/macmini-exact-clock/manifest.json`.

This is a source-hydraulic reproduction subgate. It does not create absolute
vascular blood volume from storage displacement, species or dilution state,
anatomical organ or vessel mechanics, tissue exchange, material data,
subject/activation calibration, force or temporal convergence, or sustained
standing, recovery, or walking evidence.
