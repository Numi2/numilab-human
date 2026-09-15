# Subject-scaled long-horizon native failure — 2026-09-15

The immutable receipt at
[`Docs/media/native-subject-scaled-long-horizon-failure-20260915/receipt-v3.json`](media/native-subject-scaled-long-horizon-failure-20260915/receipt-v3.json)
retains two physical Mac mini attempts using the subject-scaled `NHRIGID2`
input (`65.50000002491288 kg`) and the public `NHMYO2` muscle, tendon,
support, and equality payloads at the canonical `12.5 us` clock.

The 128-step run without `MTL_DEBUG_LAYER` and the 512-step run with the
validation layer both admitted tendon, equality, and support payloads, then
stopped before the native result line. Samples identify the main thread in
`compileStaticStandActivation` and
`MetalArticulatedOperatorSubmission::wait`/`waitUntilCompleted`. The agent
terminated each run after the observed wait interval and retained stdout,
stderr, and the sample report.

A one-step control with the same build and inputs completed in the same native
path. It reached one exact-clock step, 128 dynamic audit rows, bitwise trace,
six active supports, zero penetration, `0.146436423063 m/s2` peak acceleration,
and `0.03299692183 N` dynamic residual. The blocker is therefore the
multi-step persistence path, not input admission or first-step execution.

This is a native execution blocker, not a mechanics result. The only completed
subject-scaled native result remains the bounded 64-step replay. Long-horizon
force convergence, sustained standing, recovery, walking, activation
calibration, anatomical loading, material calibration, organ/blood/tissue/fat
ownership, and subject prediction remain unqualified.
