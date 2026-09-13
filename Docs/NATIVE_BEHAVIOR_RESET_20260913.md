# Native Human reset behavior measurement

The native accepted-root metric owner now measures the exact reset pose on the
first begin-step command buffer before any physical step. The measurement uses
the same source-bound `NHBHV1` criteria, cooked body-frame rebasing and paired
body/Jacobian/velocity buffers as accepted-root samples. A small Metal seed
kernel records only `initial_posture_valid` and `initial_settled`; it cannot
advance time, increment accepted-root counters or authorize publication.

The native snapshot now identifies a successfully compiled source-bound metric
program as `generic_taskpack_lowering=source_bound_metric_program`. This label
means that the NHBHV1 criteria were compiled against the actual native body
mapping, clock and cooked COM offsets. It is not the generic TaskPack lowering
receipt required by the frozen behavior-evidence evaluator.

## Physical Mac mini check

The isolated native checkout was `8d55506c5eabd5bc75ba70b9acd938e22b7ef5e5`
on an Apple M4 Pro. The rebuilt artifacts were:

```text
libmetalrobo.dylib  ca6a1fd9034982ab1136dcf26c988228f3253f36b9f77d236f8eee423053a084
MetalRobo.metallib  de5f667fa8fa31839a63f7c630bed084fe795b9a4fcacb25277a6802d17e39f5
```

The native target and its CTest registrations passed:

```text
human_behavior_telemetry=pass controls=10 pending_publication=denied committed=once early_reject=preserved retry=pass checkpoint_replay=bitwise double_flush=bitwise nonzero_epoch=pass mismatched_reset=denied paired_threshold=pass audit_coverage=unknown physical_steps=0
human_behavior_program=pass positive=4 negative=23 source_origin_rebase=pass partial_coverage=explicit exact_ns=pass physical_steps=0
2/2 tests passed: numi_human_behavior_telemetry, numi_human_behavior_program
```

The Swift snapshot decoder and regression fixture were updated in Brain commit
`2961d1c`. The reset observation is now available to a real producer run, but
this check contains no physical Human step and therefore does not qualify
standing, recovery, walking, anatomy, materials, calibration or the frozen
single-male capability protocol. Forbidden-contact and native force-audit
coverage remain explicitly unavailable and `full_behavior_qualified` remains
false.
