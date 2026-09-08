# Source-compliant dynamic joint limits

The coupled NumanX/Matter runtime now owns all 122 scalar source limits,
including six knee translations omitted by the older hard-range flags.
MuJoCo source-law checks, Metal force/tangent checks and bounded Brain/NumanX
transactions pass. Loaded stance integration, sustained standing/walking and
the costal timeout remain open.

Native source is `349f8b573dfa02bd9f6af7921b72b47416c671da` on `coupled`.
Brain source is `a885ff2868199c2ee2d6c61069668d56963fe90f`, including the current
NumiLab transport upstream. Human compiler/oracle source is
`59e434ad0052e8f688b3ea9117cfd441006811da`.
The [immutable receipt](media/dynamic-limits-20260908/receipt.json) records
commands, input/binary hashes, clean revisions, results and retained failures.

NHLIM1 retains the authored lower/upper range, margin, solref, solimp,
source inverse weight, REFSAFE policy and exact source/native coordinate map.
Each side activates only inside its source margin. The compliant unilateral
force enters Matter's existing Newton residual; its current active derivative
enters the same FGMRES operator. Positive rank updates add an SPD envelope to
the existing equality preconditioner. A released row leaves the exact operator
even though the envelope can retain it. No physical q/v projection, hidden
root wrench, additional controller or host physics loop is introduced.

Native configuration v6 adds mandatory immutable NHLIM1 admission to the
authored NHEQ2 world, with an optional complete costal ownership group. It
folds both source programs into the runtime identity. Brain and Gate C expose
the same descriptor; unsupported or malformed source programs fail closed.
Anatomy now exposes the actual compliant ranges and retains exact source reset
coordinates, including the six tiny out-of-range knee defaults. Legacy hard
topology retains its strict reset validation and serialization.

| Check | Result | Boundary |
|---|---|---|
| Independent MuJoCo 3.12 oracle | 101 candidates across 15 cases | Lower/upper/both sides, strict margins, release, solref/solimp and REFSAFE |
| Metal force/operator | Source force and tangent checks; 79 finite differences; factor/preconditioner checks | Exact current unilateral tangent; explicitly approximate SPD envelope |
| Failure/replay primitives | Source q/v unchanged; exact replay; invalid-pose environment isolation | Kernel fixtures |
| Native regression | 4/4 CTest checks | Limits, equality, static support and muscle/sensor transaction |
| Native admission | Eight mutated programs rejected | Fingerprint, source/policy, coordinate, padding and inverse weight |
| Anatomy | All 122 source ranges transported; exact out-of-range defaults retained | Source metadata and typed topology |
| Joint runtime | Two clean identical runs, eight accepted states each with exact physical-token replay and existing rejection/rollback checks | Three small pelvis-attached FEM samples, 12 attachments, 0.8 ms; legacy support witnesses |
| Human compiler | 201 tests, seven skips; 19 focused joint tests | Source/compiler checks, not anatomical mechanics |
| Brain regression | 13 selected tests, one skip | Descriptor, topology, locomotor program and robot-interface compatibility |

One extreme REFSAFE-off case has a velocity release threshold around 8,617.
FP32 subtraction near that threshold produced 0.000718 N·s impulse error,
which failed the first test's result-only tolerance. The retained final check
uses an explicit four-epsilon bound on the subtracted operands in addition to
the unchanged 5e-4 scaled source tolerance. Ill-conditioned finite differences
are excluded using an explicit rounding bound; direct source force/tangent
checks still cover every candidate. This does not qualify arbitrary stiff
settings for full-body timestep stability.

The costal attempt was stopped after a broader process inspection found a
concurrent NumiVivo `md-run` cohort that the earlier `md-benchmark` filter had
missed. Its interrupted log is retained. Neither the existing 60-second
callback deadline nor the material, risk or convergence settings changed.
The two earlier costal timeouts remain unresolved, and contention is not an
established explanation. The concurrent workload also prevents performance
claims from the small conformance runs.

The next physical gate is the identical recruited stance, source-compliant
NHEQ2/NHLIM1 dynamics, curved NHCNT2 support and registered tissue in one
accepted transaction. The prior internal residual is still 0.591493 against
the required 0.05. The 6.4 ms legacy NHEQ1 visual-horizon contact/limit failures
have not been reclassified as successes. Follow the
[completion plan](STANDING_WALKING_COMPLETION_PLAN_20260908.md) for loaded
source/timestep convergence, accepted-root behavior metrics, causal standing
and walking, and held-out calibration.

Recheck the stored bundle without a GPU:

```sh
python3 Docs/media/dynamic-limits-20260908/verify_receipt.py
python3 Docs/media/dynamic-limits-20260908/test_verify_receipt.py
```

Regenerate the independent source fixtures with the pinned MuJoCo 3.12 source
environment using `tools/qualify_joint_limit_source.py --output FILE`; the
native `numanx.physics.human_source_limits` CTest includes the frozen fixture.
The [source constraint implementation](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_core_constraint.c)
is the oracle authority. This is simulation conformance, not experimental
material calibration or full Human completion.
