# Accepted-step standing diagnostic — 2026-10-02

Internal engineering evidence for the standing and anatomy gap audit. This is not a board deliverable or a standing qualification.

## Provenance and repeatability

The native Apple Metal persistent stand completed 512 steps at 1 ms with 16 contact iterations and exit code 0. The six input payload hashes and full command are recorded in `command.json`; they match the earlier uninstrumented 16-iteration run. The trace build adds an opt-in accepted-state copy, enabled with `NUMI_HUMAN_STAND_ACCEPTED_TRACE=1`.

The accepted trace contains 512 rows for the requested horizon. Two preceding one-step parity transactions also emitted a row; those two rows are excluded from the horizon analysis. All accepted q, v, and generalized-force values are finite. The traced and untraced 16-iteration runs match exactly on completed steps, acceleration peak and location, maximum penetration, terminal impulse sample, maximum velocity/configuration change, equality residuals, and balance status. The solver-reported Metal time changed from 76.887 s to 76.958 s.

Binary SHA-256: `116dbe411e043d2a216c5b2feefa0f28865a5e18a04a9ff262de456e3c30c00e`
Metal library SHA-256: `a49bcefddc9804e7b15fae2f24f840dbaab55c4f507d00c6afee0288d8683785`

## What the trace establishes

- The 16-iteration run is not stable standing: `compiled_stand_balanced=false`.
- The largest independent acceleration predictor is 3,396.17 rad/s² at step 459 on DOF 37. Accepted finite-difference acceleration there is 3,333.94 rad/s². The source-joint manifest maps DOF 37 to `shoulder_rot_r`; its velocity reaches 3.56 rad/s and generalized muscle force is 23.66 N·m at that step.
- The next largest accepted finite-difference acceleration is 3,037.35 rad/s² on DOF 75 at step 230. The manifest maps it to `shoulder_rot_l`.
- Right knee flexion (`q[106]`, `knee_angle_r`) moves from 0 to 0.91920 rad (52.7°) over 512 ms. Thus the motion failure is whole-body; the peak acceleration is in the shoulders, while the knee also drifts substantially.
- Maximum support penetration is 1.813 mm. The reported maximum joint-equality position error is 57.1 µm and velocity error is 0.0571 m/s.
- The run uses uniform activation 0.8 on all 416 source muscles and has no closed-loop balance controller. It is a bounded dynamics diagnostic, not a valid 10-second standing rollout.

The status ABI overwrites `contactAndAcceleration.z` with the current step's summed normal impulse; it is not a horizon-cumulative value. The x/y/w status fields are running min/max values. Treat the impulse trace as per-step samples and do not sum or label the final sample as total horizon impulse without a separate audit.

## Iteration sensitivity

The matched 64-iteration run also fails balance. It reports 1,499.11 rad/s² at step 295 on DOF 77, 1.811 mm maximum penetration, 13.0 mm/s maximum equality velocity error, and 0.92953 rad maximum configuration change. The 16-iteration horizon instead peaks at 3,396.17 rad/s², 1.813 mm penetration, 57.1 mm/s equality velocity error, and 0.91920 rad configuration change. Contact iteration count materially changes the trajectory but does not produce a qualified stand or remove the knee drift; do not treat a single iteration setting as a fix.

## Remaining source/anatomy gaps

The left `knee_angle_translation2_l` source law remains inconsistent with its authored `[0, 0.006792] m` range; the pinned law yields a dependent value of `-4.273 mm` at `knee_angle_l=0.9 rad`. This needs the source range/law owner corrected before changing limits or pose evidence.

The patella anteriority diagnostic still passes for all 16 compiled, equality-projected poses (minimum anterior distance 6.093 mm). That establishes front/back orientation for those poses only. Cartilage contact, tracking through loaded flexion, and anatomy/clinical qualification remain open.

Next useful gate: repair the left-knee source law/range contract, then add calibrated, feedback-controlled recruitment and rerun a short accepted-state trajectory audit before extending duration. Keep the board video and other board outputs paused until requested.
