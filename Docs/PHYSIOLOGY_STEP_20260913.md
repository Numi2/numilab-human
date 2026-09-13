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

This is an engineering subgate for conservation and accepted-step history. It
does not provide anatomical vessel tubes, blood density, cardiac activation,
organ mechanics, material calibration, subject calibration, or sustained
standing/walking evidence. The systemic-physiology and organ/blood completion
statuses therefore remain unchanged.
