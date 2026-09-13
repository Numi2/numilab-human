# Human behavior telemetry exact-clock receipt — 2026-09-13

The native Human behavior compiler and telemetry qualification now use the
same canonical `12,500 ns` control quantum as the additive native ABI8/v2
exact-clock runtime. The shared C++ constant is
`kNumiHumanBehaviorTimestepNanoseconds`; the telemetry candidate timestamps,
release timestamps, checkpoint clock assertions, and retry assertions all
derive from it. The program checker also treats a non-12,500 ns binding as a
negative case.

Native source was tested on the physical Apple M4 Pro from branch
`human-blood-mass-20260913` at commit
`ac3b4af73c6ae8efef6f07d6e626e71b52e68c75` (tree
`4d98915adff9a43964a7e77088212f52953966e7`). The focused selection passed
9/9:

```text
numi_human_behavior_telemetry
numi_human_behavior_program
numanx.integration.human_matter_exact_candidate
numanx.integration.fullbody_bridge_prepared_root
numi_human_tissue_mass_partition
matter.compiler.vascular
matter.metal.vascular
matter.compiler.vascular_human_binding
matter.metal.vascular_cavity
```

The direct receipts report:

```text
human_behavior_program=pass ... exact_ns=pass physical_steps=0
human_behavior_telemetry=pass ... audit_coverage=unknown clock=12500ns physical_steps=0
```

The Brain decoder is bound to the same contract at `numi-brain` commit
`8f0c3a048d7f11b0466fae1a26077ff10d281c30`: noncanonical metric steps are
rejected, the 12,500 ns fixture is accepted, and the focused
`MetalNumanXBehaviorTelemetryTests` suite passes 17/17. This is a fail-closed
receipt parser boundary; it does not change the Brain C ABI's existing
microsecond timestamp fields or claim a completed Brain-to-native exact-clock
transaction.

The raw Mac mini evidence is retained in
[`native-current/`](media/human-behavior-clock-20260913/native-current/):
[direct output](media/human-behavior-clock-20260913/native-current/direct.log),
[build log](media/human-behavior-clock-20260913/native-current/build.log),
[focused CTest log](media/human-behavior-clock-20260913/native-current/ctest.log),
and their SHA-256 values in
[`SHA256SUMS`](media/human-behavior-clock-20260913/native-current/SHA256SUMS).

This closes the clock-unit inconsistency in the behavior compiler/telemetry
engineering path. The telemetry fixture remains a one-body synthetic
measurement with `physical_steps=0` and `audit_coverage=unknown`; it does not
qualify force or temporal convergence, anatomical supports/loading, source
activation calibration, unresolved materials, subject calibration, sustained
standing, sustained walking, or full release.
