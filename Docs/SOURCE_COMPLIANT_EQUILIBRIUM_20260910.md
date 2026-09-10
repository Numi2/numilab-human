# Source-compliant equilibrium preparation

Native `598d2b14d95ec7bef4f12492b89c3e3a54cb057c` closes the offline
source-compliant equilibrium blocker. This is preparation of a force-balanced
initial condition. Dynamic settling, registered anatomical tissue and
experimental calibration remain separate release gates.

## Implementation and physical boundary

`NumiHumanCompliantEquilibrium` solves the full 128-coordinate force balance
using the authored NHEQ2 equality and NHLIM1 stop laws. At rest their signed
force is `a_ref / R`, with source `solref`, `solimp`, REFSAFE and authored
`dof_invweight0`. Dependent coordinates can deform under load. The compiler
introduces no ideal equality multiplier, hard stop reaction or root assistance.

The native FP64 search jointly adjusts scalar coordinates, bounded muscle
recruitment and nonnegative forces at the initially touching curved support
witnesses. Every candidate reevaluates exact muscle paths, stationary fibre
balance, joint laws and support geometry. Accepted search steps reduce the
objective; final admission recomputes the mass-normalized force residual at the
final pose. The unchanged acceptance bounds are 0.05 for the maximum coordinate
balance acceleration and one micrometre for loaded support geometry. Angular
and translational coordinates have different units; their maximum is not a
linear acceleration in metres per second squared.

Search iterations are offline preparation, not physical time. The diagnostic
CLI preserves finite unsuccessful candidates with exit code 2 and rejects
malformed inputs with exit code 1. The runtime consumes only the existing
NHINIT1 pose/muscle state and authored packs; no host force correction is
injected into physical stepping.

## Results on the physical Mac mini

| Check | Result | Interpretation |
|---|---:|---|
| Old ideal-projected state under actual source compliance | Maximum balance acceleration 8,823.874 | Rejected initial condition |
| Joint pose/recruitment solve | Maximum 5.50121e-7; RMS 1.37543e-7 | Full FP64 stationary force sum passes |
| Loaded witness gap | Below 1.1e-16 m in the native FP64 calculation | Geometric admission passes |
| Independent clean compile | Entire result bytewise identical | Offline replay passes |
| Fixed FP32 pose and recruitment, support sharing re-solved | Maximum 0.00804238; RMS 0.00139031 | Passes the unchanged 0.05 bound |
| Same FP32 support check | Maximum loaded gap 2.81118e-8 m | Passes the unchanged geometry bound |
| Source scalar law and native support regressions | 3/3 pass with Metal API validation | Analytic, source-fixture and Metal coefficient checks |
| Four-root Brain/native comparison | 0.4 ms per scenario; exact replay and dropout/zero motion | Bounded physical transport passes |

The FP32 check fixes the actual stored pose and activation bytes and re-solves
stationary fibres in FP64. It does not certify force from the frozen quantized
fibre lengths by itself. The physical transactions consume those admitted fibre
bytes. A zero-iteration FP32 diagnostic failed because floating-root wrench
sharing alone does not determine the internal foot load distribution; the
support-only solve fixes that distribution without changing pose or activation.
Both results are retained.

The static solution uses 237 nonzero activations, including 47 at the upper
bound, and some coordinates reach their allowed 0.15 displacement bound. These
are optimizer outputs, not measured or physiologically calibrated recruitment.
The Matter package still contains three small pelvis samples and 12 attachments;
it does not register the full anatomical tissue stack.

An independent MuJoCo 3.12.0 forward calculation verifies all 57 model source
files against the pinned source archive. It compares mass action, gravity,
curved support geometry and the support wrench, without stepping dynamics.
Maximum relative mass-action error is 9.91e-14, gravity error 9.25e-7, support
lowering error 3.42e-9 m, and vertical wrench residual 2.90e-5 N. Source scalar
law conformance is covered separately by the native equality/limit checks.
An initial array-library warning log remains retained; explicit finite-checked
contractions reproduce the same source audit without warnings. Its cause has
not been established.

## Dynamic evidence and remaining blocker

In the four-root comparison the recruited right ankle reaches 0.0101566 rad/s
and root linear speed 0.00226358 m/s, compared with 0.0101230 rad/s and
0.00232290 m/s under zero commands. Maximum delivered excitation is 0.203731;
terminal activation differs by 0.00300264 and signed applied muscle force by
0.129868 N. This improves the initial motion relative to the earlier prepared
state, but does not establish held stance.

The 64-root cohort completes 6.4 ms per scenario and the recruited trajectory
replays exactly. Recruited root speed reaches 0.0347815 m/s. Independent trace
verification rejects the dropout/zero comparison: its kinematics differ from
the first root despite identical muscle commands and activation. The original
Swift test reported success because it checked dropout commands but omitted
physical equality. Brain `c7db5de` adds the missing physical assertions and
separates signed applied force from positive tendon tension in trace v2. This longer cohort is retained as failed comparison evidence;
it cannot promote dynamic balance or reset reproducibility.

A fresh four-root cohort on clean published native `9897bf8` and Brain
`c7db5de` reproduces exact dropout/zero state with the stronger assertions.
The instrumented long cohort reproduces the original failure and the test now
returns failure. Native `9897bf8` supplies optional first-root physical snapshots.
They prove identical source pose, muscle force, bias and free velocity across
all four scenarios. The first measured difference is in the force after Matter's
reaction is added: its maximum difference is 1,691.1521, on root translation Z
(in newtons). This confines investigation to the coupled solve, without
establishing its internal cause. See [the buffer comparison](media/source-compliant-equilibrium-20260910/initial-buffer-comparison.json). Controller gains and protective inhibition remain unchanged.

The first long attempt also exhausted its 300-second runner deadline. The later
600-second runner completes in 444.2 seconds, with one second of sampling during
execution. This is not a performance qualification. The sample finds the host
waiting for GPU submission/completion and does not identify a dominant kernel.
Failed compilation attempts, deadline failure and contradictory traces are
retained in the evidence bundle.

The same-state source/native path check finds two outliers: left `EDC5_l`
differs by 0.274851 mm and `EDC4_l` by 0.174194 mm. The next largest error
among the other routes is 0.576154 micrometres. All initial path velocities
are zero. These outliers coincide with the largest initial muscle-force
changes and identify a concrete wrapping-geometry check; causation and repair
remain to be verified. See [the path audit](media/source-compliant-equilibrium-20260910/initial-path-audit.json).

## Next completion gates

1. Trace and repair the divergent Matter reaction from identical source inputs; require exact
   reset/replay/dropout state across the longer cohort before controller tuning.
2. Verify force/path/fibre parity at the identical admitted FP32 state, then
   source/timestep convergence and held mechanical stance.
3. Register anatomical tissue at that pose, with conserved mass, prestress and
   force/moment transfer; resolve loaded cartilage/meniscus and LCL dependencies.
4. Implement accepted-root behavior telemetry and causal stance/recovery/gait
   control through the existing Brain owners, preserving protective inhibition.
5. Execute the frozen 420 behavior trials, independent held-out calibration,
   mesh/time/parameter uncertainty checks and five Apple performance workloads.

The [standing/walking plan](STANDING_WALKING_COMPLETION_PLAN_20260908.md) and
[release matrix](NEUROMUSCULOSKELETAL_RELEASE_MATRIX.md) retain the full gates.
See [the receipt](media/source-compliant-equilibrium-20260910/receipt.json),
[local verifier](media/source-compliant-equilibrium-20260910/verify_receipt.py)
and [independent source audit](media/source-compliant-equilibrium-20260910/verify_source.py).
