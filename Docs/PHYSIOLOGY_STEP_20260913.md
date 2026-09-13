# Accepted physiology-step owner — 2026-09-13

The Human physiology authoring layer now has a bounded accepted-step owner in
`numilab_human.physiology_step`. It advances a compiled circulation graph with
semi-implicit edge flow, conservative upwind species transport, declared
blood-to-tissue exchange, positive-volume/amount checks, and deterministic
candidate rollback. Rejected candidates do not advance accepted time or alter
the accepted-state trace.

The retained [receipt](media/physiology-step-20260913/receipt.json) executes the
source-locked `passive_transport_fixture` for 32 attempted steps at `100 µs`.
One candidate is deliberately rejected; 31 accepted roots preserve total
compartment volume to `8.47e-22 m³` and the synthetic tracer amount to
`1.27e-21 mol`. The output is `fixture_only`, with an immutable graph hash and
accepted-state trace hash.

The command is available through both interfaces:

```sh
numi human physiology-step \
  --graph config/physiology-passive-fixture.v1.json \
  --sources Sources --steps 32 --timestep-seconds 0.0001 \
  --reject-step 9 --output /absolute/path/receipt.json
```

The same owner was re-run on the physical Mac mini from commit
`90b085648b624c5259ccca1ef3eab71bfa6d42e6` in an isolated worktree. The focused
source-locked suite passed `46` tests, and the CLI receipt accepted `15` of
`16` attempts with one deliberate rollback; both volume and tracer amount
remain conserved. The retained command, logs, receipt, and SHA-256 manifest
are in [`media/physiology-step-20260913/macmini/`](media/physiology-step-20260913/macmini/).

This is an engineering subgate for conservation and accepted-step history. It
does not provide anatomical vessel tubes, blood density, cardiac activation,
organ mechanics, material calibration, subject calibration, or sustained
standing/walking evidence. The systemic-physiology and organ/blood completion
statuses therefore remain unchanged.
