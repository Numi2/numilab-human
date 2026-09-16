# Canonical Human trace comparison, 16 September 2026

This directory contains a reproduction script, not a new runtime or qualification gate.
The source-locked runs use `Numi2/numi-lab` workflow run `35150128058` at native
revision `f933434ef94fb9de2afd2b6706a94d049a70a152`, and the canonical Human
launcher and source payloads at `21f27ccbd6562022103267bde6f2e7ef680eea9d`.

Extract the four `human-source-locked-release-{100us,50us,25us,12p5us}` Actions
artifacts into four separate directories. Each directory must retain its own
`execution.json` and named case subdirectory. Then run:

```sh
python3 compare_traces.py /path/100us /path/50us /path/25us /path/12p5us
```

The script requires all four complete, source-consistent 6.4 ms traces. It
compares the captured root translation, quaternion orientation, internal
configuration, generalized velocity and interval-integrated normal impulses
at 65 shared timestamps. It does not compare all authoritative low components,
muscle states, complete vector reactions or impulsive work. Average support
load and similar trajectories alone cannot establish force convergence or
sustained standing. Execution uses a hosted Apple Paravirtual device, not a
physical M4/M4 Pro. No new numerical acceptance threshold is inferred from
these runs.
