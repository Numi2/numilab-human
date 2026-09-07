# Frozen Human behavior evidence v1

`numi human behavior-qualify` (also available as
`python -m numilab_human.behavior_qualification`) admits a complete, immutable
behavior evidence bundle against an independently supplied protocol and current
stack. It does not execute a controller or promote the existing millisecond
Human/Matter demonstrations into standing or walking evidence. **The native
accepted-root telemetry producer described below is not integrated yet. No
Human capability is qualified by this tooling change.**

The fixed population gates are:

| Task | Frozen population | Full accepted horizon per trial | Outcome gate |
| --- | --- | --- | --- |
| Standing | 20 distinct seeds | 60 seconds | All 20 remain settled with no posture violation |
| Recovery | 100 distinct seeds and nonzero frozen impulses | 5 seconds after the impulse | At least 95 regain the frozen settled condition and maintain it for the protocol's hold duration, with no posture violation |
| Walking | 100 distinct seeds at each of 0.5, 1.0 and 1.5 m/s | 120 seconds | At least 95 trials **at each speed** have no posture violation and forward-speed RMSE at most 0.15 m/s |

A failed task outcome remains in its population denominator. A missing,
truncated, assisted, duplicated or otherwise invalid trace invalidates the
bundle; it cannot be counted as one of the five permitted task failures. These
are behavioral gates only, separate from anatomical, material, source-parity,
replay and performance qualification.

## Frozen inputs and current identity

The caller must supply `--protocol-sha256` from the protocol frozen before
execution and `--stack-sha256` from the current stack authority. The evaluator
does not adopt either identity from a historical evidence bundle. Changing any
protocol criterion, reset state, impulse, seed, task, speed, stack revision or
artifact creates a different qualification target.

```sh
PYTHONPATH=src python3 -m numilab_human.behavior_qualification \
  --protocol /absolute/path/protocol.json \
  --protocol-sha256 "$FROZEN_PROTOCOL_SHA256" \
  --stack /absolute/path/current-stack.json \
  --stack-sha256 "$CURRENT_STACK_SHA256" \
  --bundle /absolute/path/behavior-bundle.json \
  --output /absolute/path/new-assessment.json
```

`schemas/human-behavior-evidence-v1.schema.json` defines the structural wire
schema for the protocol, stack, bundle and individual JSONL trace records. The
Python validator additionally checks relations that JSON Schema cannot express:
populations, unique seeds and impulses, current artifact bytes, transaction
continuity, duration, reductions, and per-task outcomes.

The stack identifies clean Human, native runtime and Brain source revisions,
Apple device and OS build, and exact bytes of HumanPack, CompiledRun, TaskPack,
PolicyPack, runner, native library, Human/Matter/Brain metallibs, evaluator,
compiled native metric program, accepted-root proof schema and TaskPack lowering
receipt.
Additional artifact roles are permitted and are also verified. Paths are
relative to the manifest that contains them, or absolute. Required files must
be retained and readable. The evaluator hashes all artifacts before admission
and again after reading the traces, including its own executable source. A
matching revision label alone is insufficient.

The protocol binds the exact TaskPack bytes and uses integer nanoseconds for
its fixed physical step. That step must exactly divide all three fixed
horizons. Recovery's hold duration must cover whole steps, be positive and fit
inside its five-second horizon. It also registers minimum root height, maximum
trunk tilt, semantic forbidden-contact IDs and, for standing/recovery, maximum
planar speed. These criteria need physical review and must be lowered into the
TaskPack before a native producer may assert the reduction contract. The
evaluator does not invent universal anatomical or balance thresholds.

Every task must identify the source-semantic root and trunk bodies, world
reference origin, orthogonal unit world up/forward axes, trunk's local up axis,
and velocity observable. Height is the root body's world origin relative to the
reference origin, projected onto world up. Trunk tilt compares its transformed
local up axis against world up. Velocity is explicitly either a named body's
COM velocity or the whole-Human COM velocity; the latter has no single body
owner. Planar speed excludes the up component; forward speed projects onto the
frozen forward axis. The native compiler must resolve these semantic IDs and
observables against the actual HumanPack and retain the lowered metric program.

`task_lowering_receipt` is a hashed stack artifact with schema
`numi.human.behavior-task-lowering.v1`. It binds the frozen protocol hash, exact
criteria, physical step, native source revision, and hashes of HumanPack,
CompiledRun, TaskPack, native library, all three metallibs, compiled metric
program and accepted-root proof schema. The evaluator checks every one of those
bindings against the current stack before reading trials. A merely hashed
TaskPack without this source-bound lowering receipt cannot be admitted. The
native compiler/runner must produce and validate this receipt; its production
is part of the integration work still outstanding.

## Native producer contract

Each trial is one UTF-8 JSONL file: one `numi.human.behavior-trial.v1` header,
one or more `accepted_span` records, then exactly one `completed` footer. Every
line ends in a newline. The bundle records each complete file's SHA-256 and byte
length. Duplicate JSON keys, nonfinite numbers, boolean counters, duplicate
trial IDs, reused execution IDs and reused trace bytes or paths are rejected.
Frozen numeric fields never accept JSON booleans as equivalent values. Impulse
descriptors admit only the five physical fields; metadata cannot distinguish
duplicate physical perturbations, and numeric spellings such as `1`/`1.0` or
`0.0`/`-0.0` do not make distinct impulses.

The header repeats the exact frozen case and protocol/stack hashes, actual native timestep and OS
build, and hashes of the admitted metric program, proof schema and lowering
receipt, plus native
`initial_posture_valid` and `initial_settled` observations. Standing requires a
settled reset, and every task requires a posture-valid reset. The native
runner must attest the actual reset state, selected task, seed, device and
applied impulse, rather than copying unverified command-line expectations.
Recovery applies its source-bound world-frame impulse at reset, before the
first accepted root. The impulse's body, point and vector must be retained in
the frozen case. Rejecting a candidate must not apply the impulse twice.

The Metal owner must reduce metrics **only after a complete Brain–Human–Matter
root is accepted**, over every physical root without decimation. Bounded
aggregate spans avoid a Python or host per-step physical loop. Each span has
contiguous root indices and integer accepted-time boundaries, accepted and
rejected attempt counts, and the prior/final accepted-root hashes. Root hashes
must cover the complete accepted transaction state, not just rendered pose.
Rejected candidates contribute to attempt accounting but never to time,
outcome, speed, controller or sensor state.

Each span also reports `metric_sample_count` and `audit_covered_root_count`,
both exactly equal to accepted roots, plus `audit_covered_attempt_count`, exactly
equal to accepted and rejected attempts together. Walking additionally requires
`speed_error_sample_count` equal to accepted roots; the other tasks require
zero speed-error samples. These counters must be emitted by the respective
native reduction/audit paths, independently of the span's scheduling counts.
Missing, zero-initialized, decimated or otherwise incomplete metric coverage
invalidates the trace even when its reported squared error is zero.

The native reduction definitions are:

- `posture_violation_steps`: accepted roots at which root height is below the
  frozen minimum, trunk tilt exceeds the frozen maximum, or a registered
  forbidden contact occurs. Walking and recovery also require zero violations;
  a fall followed by an eventual upright frame is not recovery success.
- `settled_steps`: accepted roots with no posture violation and planar speed at
  or below the frozen bound for standing/recovery. For walking, this counter
  records roots without a posture violation; speed is assessed separately. The
  walking count must equal accepted roots minus posture violations; incomplete
  or contradictory reductions invalidate the trace.
- `settled_suffix_steps`: the number of consecutive settled accepted roots at
  the end of that span. The evaluator composes suffixes across spans. Recovery
  time is the first accepted sample of the final uninterrupted settled interval,
  which must last for the frozen hold duration through the end of the trial.
  For a suffix of `k` samples, that first sample is at `end - (k - 1) * step`;
  the preceding unsampled step is never counted as settled. An initially settled
  reset followed by an entirely settled trace permits a start time of zero.
- `speed_squared_error_sum_m2_per_s2`: the sum over every accepted walking root
  of `(forward_velocity_mps - target_speed_mps)^2`, using the TaskPack's frozen
  forward frame and velocity observable. With fixed physical steps, dividing
  the sum by the accepted root count gives the full-horizon mean squared error.
  Standing and recovery emit zero for this walking-only field.

Every span must also report zero for all eight admission audits:
`root_assistance_steps`, `direct_torque_steps`, `kinematic_override_steps`,
`unregistered_force_steps`, `source_constraint_omission_steps`,
`unaccepted_publications`, `nonfinite_steps` and `unexpected_reset_steps`.
Source-prescribed forces and the single frozen recovery impulse are registered
physical forces; hidden stabilization or direct generalized control torque is
assistance. The constraint audit must be driven by the actual admitted source
constraint program and accepted solve, not by a requested configuration flag.
A footer requires the exact accepted step count, final root hash and successful
native process completion. Later records and partial final lines are invalid.

The missing integration is a native accepted-root reducer, source-bound
TaskPack lowering for the frozen metric criteria, bounded Swift orchestration
of deterministic resets and all 420 cases, and complete retained native traces.
Source/build success or hand-authored summaries cannot establish that this
producer contract was executed. SHA-256 checks bind evidence bytes and reject
stale mixtures; they do not authenticate the producer or independently prove
the physical measurements.

## Assessment and verification

The evaluator writes a new report with `passed`, `failed` (complete admissible
trials miss an outcome gate), or `invalid` (missing or inconsistent evidence).
It never overwrites an earlier report. The report records each trial and
per-speed population, plus protocol, stack and bundle hashes. No output mutates
the completion ledger or promotes a capability automatically.

`tests/test_behavior_qualification.py` uses visibly synthetic artifacts and
aggregate counters solely to test this admission contract. Its successful
fixtures are not retained as Human behavior evidence. Run it with:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests \
  -p test_behavior_qualification.py -v
```
