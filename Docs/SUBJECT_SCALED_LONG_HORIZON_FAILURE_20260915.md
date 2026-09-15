# Subject-scaled long-horizon native failure — 2026-09-15

The immutable receipt at
[`Docs/media/native-subject-scaled-long-horizon-failure-20260915/receipt-v4.json`](media/native-subject-scaled-long-horizon-failure-20260915/receipt-v4.json)
retains two physical Mac mini attempts using the subject-scaled `NHRIGID2`
input (`65.50000002491288 kg`) and the public `NHMYO2` muscle, tendon,
support, and equality payloads at the canonical `12.5 us` clock.

The 128-step run without `MTL_DEBUG_LAYER` and the 512-step run with the
validation layer both admitted tendon, equality, and support payloads, then
stopped before the native result line. Samples identify the main thread in
`compileStaticStandActivation` and
`MetalArticulatedOperatorSubmission::wait`/`waitUntilCompleted`. The agent
terminated each run after the observed wait interval and retained stdout,
stderr, and the sample report; these are timeout evidence, not proven native
crashes.

One-, two-, and eight-step controls with the same build and inputs completed in
the same native path. They reached 1/2/8 exact-clock steps, 128 dynamic audit
rows, complete bitwise traces, six active supports, zero penetration, and the
same `0.146436423063 m/s2` peak acceleration and `0.03299692183 N` dynamic
residual. Native muscle-force work scales from `207.877959 ms` for one step to
`1,708.092375 ms` for eight steps. The retained blocker is therefore the
unqualified long-horizon throughput budget, not input admission or an observed
first-step/two-step state break.

This is a native execution blocker, not a mechanics result. The bounded 64-step
replay remains the only released subject-scaled qualification; the 1/2/8-step
captures are execution controls. Long-horizon
force convergence, sustained standing, recovery, walking, activation
calibration, anatomical loading, material calibration, organ/blood/tissue/fat
ownership, and subject prediction remain unqualified.
