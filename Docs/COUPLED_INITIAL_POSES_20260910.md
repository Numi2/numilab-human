# Coupled support initial-pose repair

Native `baca930c27dda11dfd6e0afd57f696955c054d48` repairs the missing initial
body-pose input to Human support recovery. It is published on `origin/coupled`.
The contact law, stabilization, tolerances and Brain protection are unchanged.

The retained failure compares four consecutive scenarios at the same prepared
source-compliant pose. The recruited trajectory replays, but the fourth
scenario, with unavailable observations and zero excitation, differs from the
zero-excitation reference at its first root. Initial q, muscle generalized
force, bias, free velocity, candidate contact points and Jacobians match. The
first differing value is support row 1's recovery constraint before the first
linear solve: `17.9370499` versus `395.966522`.

The Human adapter supplied null `rigid.currentBodies`. Support did not require
an initial-body arena, so its kernel indexed the runtime's dummy buffer as body
records. The recovery target depends on those initial poses; candidate poses
alone cannot establish it. Identical input commands therefore did not guarantee
identical physical results across scenario allocations.

The adapter now materializes the owner's existing COM poses into separate
private GPU scratch before Matter. Every record is written on the same command
buffer, with checked dimensions and retained-memory accounting. Candidate
kinematics cannot overwrite it. The explicit `humanSupportInitialBodies` input
supplies only support's initial-gap reference. Generic MetalWorld callers keep
their existing current-body input. Missing, truncated, wrong-device and
insufficiently covering inputs fail before support encoding. No host physics or
additional body dynamics authority was introduced.

Ten native checks pass with Metal API validation: support load/friction/replay,
support linearization, static support, source equality/limits, Human/Matter
owner/v4/candidate/adapter and attachment runtime. The support test additionally
rejects a missing initial arena and both implicit and explicit arenas truncated
by one byte. The adapter's allocation loop did not reproduce Objective-C address
reuse. It now reports that fact separately and continues to check ACCEPT,
REJECT, stale generations, rollback and publication. Pointer-reuse coverage
remains partial; the earlier failed test log is retained.

The repaired four-root comparison passes bitwise replay and dropout/zero
physical equality across all ten trace kinds. It covers 0.4 ms per scenario and
shows delivered excitation changing activation and muscle force. The exact
published-stack 64-root qualification also passes: all four scenarios complete
6.4 ms, recruited replay and dropout/zero equality are bitwise across all ten
trace kinds, and all six first-root physical snapshots match. The unchanged
validation-enabled runner took 445.486 seconds for 256 accepted roots. Terminal
recruited root speed is 0.0347784 m/s; this remains unheld motion.

Evidence and verifier: [receipt](media/coupled-initial-poses-20260910/receipt.json)
and [verification script](media/coupled-initial-poses-20260910/verify_receipt.py).
The failed 64-root contact-assembly trace, diagnostic source patch, failed test
attempts and repaired results remain separate artifacts.

This closes initial-pose ownership for bounded support transactions. It does
not establish held stance, walking, registered anatomical tissue, experimental
calibration or physiological-duration convergence. The two prepared wrist-route
outliers, full fibre/force parity, tissue registration and practical execution
cost remain on the critical path.
