# Numi Human cap-8 performance qualification preregistration

Status: `incomplete`

This contract converts the existing single-sample cap-8 shader timing
observation into an exact-workload performance receipt. It does not qualify
standing, force convergence, whole-Human behavior, or any of the five broader
`PerformanceEnvelope` rows.

## Frozen comparison

- Physical runner: the dedicated Apple M4 Pro Mac mini reached through
  `ssh macmini`; hosted or virtual Metal cannot substitute.
- Native comparison HEAD: `61c14e0a72a8a3a31a4250020c17c68ae5566020`
  for both variants.
- Baseline: the clean HEAD `src/metal/NumiHumanStand.metal`, SHA-256
  `2a33f080920bc0b26fb1528d13f0fc969b5815e7ee82b9352d740063df9f1678`.
- Candidate: only the qualified shader patch, SHA-256
  `93ce40a8a0d7c6c6e3269c1b8e33dd1e02372f2d2e93856c6e4ae48469ac7532`.
- Human inputs commit:
  `f53c2e1053bc90e7ec44fd69a5b49f96b71b71ab`.
- Workload: 512 steps at 12.5 microseconds, authoritative submission cap 8,
  64 contact iterations, dimension 640, mechanics-only, passive joint tissue
  enabled, root assistance absent, Metal debug layer disabled, and the same
  stance contacts, stance DoFs, binary, toolchain, build configuration, runtime
  environment, and unrelated Matter metallib across variants.
- Timing cases omit deterministic replay so the measured path is the production
  path. Existing replay-on equivalence remains a prerequisite, and every timed
  A/B pair must independently preserve terminal and mechanics equivalence.

Each variant has an isolated worktree and build directory. Both immutable
artifacts are built and source-bound before measurement. No compilation,
metallib rebuild, Instruments attachment, or trace export occurs inside the
timing loop.

## Fixed execution design

Let `A` be the baseline and `B` the candidate.

1. Reject readiness unless AC power, machine/OS identity, console state, free
   capacity, environment, and artifact hashes match the campaign manifest; no
   competing Numi, Metal, Xcode, build, profiling, or training process may be
   active.
2. Execute one unmeasured full-workload warmup per variant.
3. Begin a measured block only after three consecutive host samples report
   nominal thermal pressure, stable swap, low background load, and no competing
   job. Do not impose or claim a calendar-based cooldown.
4. Execute six paired blocks in this immutable order:

   ```text
   AB, BA, AB, BA, AB, BA
   ```

5. Run the two members of each block back-to-back. Retain every attempt. A
   timeout, failed contract, monitor loss, instrumentation gap, or unexpected
   competing process fails the campaign; no run may be silently excluded or
   replaced.
6. After the uninstrumented timing campaign, execute one exact-workload A trace
   and one B trace under Metal System Trace, followed by one strict detailed GPU
   counter capture for each variant. Instrumented runs are causal evidence and
   never enter timing statistics.

## Required measurements and artifacts

Every run retains its command, monotonic and UTC boundaries, stdout/stderr,
case receipt, terminal state, all mechanics metrics, authoritative segment
durations, process exit, accepted and failed steps, and hashes of every raw
file. `/usr/bin/time -lp` records maximum resident set, peak footprint, page
faults, and process swaps. Pre-run and post-run snapshots retain:

- `pmset -g therm`;
- `sysctl vm.swapusage`;
- `vm_stat` and `memory_pressure -Q`;
- `uptime` and the full process inventory; and
- machine, OS, power-source, console-user, disk, binary, metallib, source,
  input, toolchain, configuration, and controlled-environment identities.

A per-run privileged `powermetrics` capture must retain raw CPU/GPU power,
thermal, frequency, process-energy, and process-GPU samples plus its flushed
usage summary. Integrated same-device energy supplies accepted steps per joule.

The two profiling variants retain the original `.trace` bundles and exported
Metal GPU intervals, counter metadata, detailed counter tables, device thermal
intervals, and command-buffer error rows. An empty, unsupported, timeline-only,
or RT-unit-only counter result is missing evidence, not a passing substitute.

The campaign publishes immutable per-run receipts, a canonical aggregate
`numi.human.optimization-performance.v1` receipt, and `SHA256SUMS`. The aggregate
contains the preregistered order and thresholds, all twelve run IDs, paired
ratios, raw-value summaries, exact sign test, paired log-ratio confidence
intervals, energy and memory comparisons, trace/counter findings, every gate,
`exclusions: []`, and an explicit claim boundary.

## Fail-closed acceptance

The exact-workload performance receipt passes only when all conditions hold:

1. All 12 measured cases complete their native and segmentation contracts with
   zero timeout, failed step, Metal error, untyped failure, or missing sample.
2. Within every A/B block, initial and terminal `q`/`v` are bitwise identical,
   every non-timing/non-work mechanics field is exact, all eight work metrics
   remain within the existing paired FP32 bound, and the native stage and
   authoritative-segment schedules match.
3. `baseline / candidate` is at least `1.25` in all six pairs for both
   end-to-end wall time and summed authoritative-segment time. Six successes
   give the preregistered one-sided sign-test probability `1/64 = 0.015625` at
   that threshold. The receipt reports the minimum, median, geometric mean,
   range, and paired-log confidence interval; it claims the measured result,
   not the earlier 1.8134x observation if they differ.
4. Candidate per-run authoritative-segment p99 is no worse in every pair. This
   is segment latency, not yet authored control-step latency.
5. Process swaps are zero, system swap does not grow, memory pressure remains
   nominal, and candidate peak footprint is no more than `1.02` times its paired
   baseline.
6. Thermal pressure remains nominal with no recorded thermal or performance
   warning. Every pair has complete, parseable energy samples, and candidate
   accepted steps per joule are at least `0.98` times baseline in every pair.
7. Both Metal System Traces contain GPU rows, equal command-buffer and segment
   schedules, lower candidate GPU elapsed time, and zero command-buffer error
   rows. Both detailed profiles contain the same nonempty occupancy, limiter,
   and bandwidth counter set with finite samples.

Any false or unavailable condition yields `status: incomplete` or
`status: failed`; the comparator must not emit `passed` by weakening a gate.
Even a passing receipt sets `performance_envelope_qualified: false` until the
five final workload rows have separately frozen and passed their composition,
capacity, accuracy, control-period, absolute-memory, and delivery budgets.

## Current instrumentation blockers

The current Mac mini permits unprivileged `pmset`, `vm_stat`,
`memory_pressure`, `sysctl`, `ps`, `iostat`, `/usr/bin/time -l`, and Metal System
Trace collection. It does not currently provide the two remaining mandatory
surfaces:

- `powermetrics` requires superuser execution, and noninteractive sudo is not
  available. Xcode Power Profiler reports that it is unsupported on macOS, so
  it cannot replace the missing energy record.
- The SSH session exposes only the Metal timestamp counter set, with stage
  sampling but no dispatch sampling. Detailed occupancy, limiter, and bandwidth
  profiling requires an active GUI Metal debugger and Profile-after-Replay.

Accordingly, this preregistration and any timing-only campaign remain
`status: incomplete` until privileged energy capture and nonempty detailed GPU
counters are retained for both variants.
