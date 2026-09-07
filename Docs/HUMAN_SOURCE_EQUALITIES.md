# Source-compliant joint equalities in the joint Human runtime

The NumanX v4 path admits all 51 scalar-joint equality rows from the pinned
MyoSim full body into Matter's coupled solve. The separate `NHEQ2` payload
retains the source quartic relations, reference coordinates, compliance,
damping, impedance and compiled inverse-weight parameters. The existing
`NHEQ1` payload retains its explicit legacy projection behavior and exact bytes.

The source pose is not exactly on those relations: its largest initial defect
is 0.05241920054 in source coordinate units. NHEQ2 keeps that pose and applies
the source compliant acceleration law. It never silently snaps the anatomy or
reinterprets the rows as rigid kinematic constraints.

## Physical ownership

At each accepted root, the private free Human predictor remains
`v_free = v0 + h A0^-1 (source_force - source_bias)`. Source equality rows are
linearized once at accepted `q0/v0`. Their positive regularizers permit exact
algebraic elimination of the row impulses into the full-dimensional Matter
residual and `J0^T R^-1 J0` operator.

Matter builds a private `B = A0 + J0^T R^-1 J0` preconditioner using positive
Cholesky updates of the immutable source A0 factor. It improves the numerical
solve without changing the physical reaction: Human receives `A0 delta_v / h`
exactly once. There is no second equality solve in Stand, no duplicate force
scatter and no host physical stepping.

The native owner documents the equations and ABI in
[Human source equalities](https://github.com/Numi2/numi-lab/blob/coupled/docs/HUMAN_SOURCE_EQUALITIES.md).
`mrnx_runtime_config_v4` requires an authored Matter world, the NHEQ2 file and
its nonzero FNV-1a64 content fingerprint. The resulting Human source identity
includes the equality program. Invalid or unavailable v4 constraints cannot
fall back to an unconstrained world. The Brain CLI accepts `--joint-equalities`
and `--joint-equality-fp` together with its authored-world arguments.

## Evidence boundaries

The retained evidence is in
[the source-equality evidence directory](media/numanx-source-equalities-20260908/).

- The independent MuJoCo 3.12.0 oracle compares 204 source rows over four states.
  Position/Jacobian error is at most 4.45e-16; reference acceleration error is
  at most 4.55e-13. Full versus eliminated KKT velocity error is 4.82e-14.
  This is source-row and algebra evidence, not whole-trajectory parity.
- The canonical Metal probe covers 51 rows in two environments and three cases,
  including replay and source-factor immutability. Inverse action relative
  error is 1.72e-6; residual finite-difference relative error is 4.76e-4.
- The native authored full-body transaction passes at its existing convergence
  tolerance, with three FGMRES iterations in the recorded root. Eight invalid
  equality configurations and twelve invalid authored worlds are rejected.

Both Brain integration cases pass with Metal API validation, including exact
rejected-root retry, timeout rejection and joint publication. The authored case
uses NHEQ2; the legacy case remains explicit. Each accepts eight roots at
100 microseconds, only 0.8 ms per case. The final artifact receipt records the
exact tested stack and launch configuration. Nine native CTests pass with Metal
validation. The strict adapter address-reuse regression passes separately on
plain Metal; validation-layer allocations did not reproduce its required address
reuse, so that failed prerequisite is retained and is not counted as a passing
validation-layer test. Neither this source program nor a bounded accepted-root demonstration
qualifies sustained posture, recovery, walking, material calibration or the
complete anatomical composition. The anatomical Matter world in the final
HumanPack still needs source-bound mass, active-force and regional attachment
replacement maps. See the [completion ledger](HUMAN_COMPLETION_GAP_LEDGER.md).

## Completion scope and behavioral admission

`numi human target-coverage` materializes the immutable cumulative target union
from the source registers and an explicitly pinned composed MyoSim export.
Every previous leaf survives subsequent materialization. Presence in the
inventory leaves physical qualification unknown; missing source inventories
remain explicit gaps. The current union has 27,801 leaves and retains all
27,737 earlier leaves. Eight previously omitted shoulder-ligament tendons and
their source routes/sites/wraps are inventoried as `not_lowered`; their mechanics
are not promoted. Three current register gaps remain: the original bimanual
MoBL inventory, complete Z-Anatomy inventory, and one malformed legacy MyoSim
XML file. The compact materialization receipt is retained here,
while the source-derived full inventory remains local under its source licenses.

`numi human behavior-qualify` checks a complete evidence bundle against an
independently frozen protocol and current artifact stack. It requires all
20 standing, 100 recovery and 300 walking trials, complete accepted-root
metrics, unassisted force audits, exact current artifact hashes and native
TaskPack lowering. The evaluator and schema exist; the native metric producer,
controller and full qualification traces remain outstanding. Synthetic test
fixtures do not qualify Human behavior. See
[frozen Human behavior evidence](HUMAN_BEHAVIOR_QUALIFICATION.md).
