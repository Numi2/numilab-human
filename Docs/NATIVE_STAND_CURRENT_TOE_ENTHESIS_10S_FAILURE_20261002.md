# Current toe-enthesis 10-second standing attempt — 2 October 2026

The physical Apple M4 Pro attempt used the exact toe-enthesis `NHTENDON3`
payload and source-path muscle feedback, with 64 contact iterations and zero
root assistance. Runtime admission resolved all 832 endpoints as 653
distributed envelopes and 179 point fallbacks. The requested 10,000 steps did
not complete: the operator stopped the run after observed loss of support and
large state divergence. The last retained accepted progress sample is at step
5,728 (5.728 simulated seconds); the process exited with signal status 143.

The trajectory was plainly unsupported by the final retained samples. Across
progress rows, root speed peaked at 37.79 m/s, reported support force ranged
from 0 to 267.7 kN, support count ranged from zero to six and was one at the
last sample, penetration reached 9.51 mm, and the muscle-excitation correction
reached its 0.2 bound. The root was 7.729 m horizontally from its first
retained sample at the last row. No root force or torque assistance was applied.

This is a failure-focused partial diagnostic, not a completed 10-second
simulation or a standing pass. The smaller 512 ms prefix remains a separate
accepted diagnostic; the previous completed 10-second run used an older tendon
payload and does not requalify this map. The execution runtime and Human command
source commits are older than the current Human default-line head. Replay,
energy closure, measured-subject calibration, anatomical foot-contact
qualification, recovery, gait, and integrated physiology remain open. No
board video or visual deliverable was produced.

The progress stream records one sample every eight 1 ms steps. Its first
recorded horizontal root drift above 10 mm occurs at 1.576 s, with all six
contacts still active and feedback correction at 0.0366. The correction first
reaches its 0.2 bound at 2.016 s while all six contacts remain; contact count
first falls to five at 2.504 s. Root speed first exceeds 1 m/s at 3.808 s with
four contacts, and support force first reaches zero at 4.824 s. This ordering
shows the drift starts before feedback saturation and contact loss, but the
sparse progress samples do not isolate the initiating mechanical or control
cause. A source-aligned failure-onset diagnosis is still needed before another
long-horizon standing attempt.

The exact command, input/output hashes, partial stdout, stderr, execution
context, and machine-readable [receipt](media/native-stand-current-toe-enthesis-feedback-10s-20261002/receipt.json)
are retained under
[`media/native-stand-current-toe-enthesis-feedback-10s-20261002/`](media/native-stand-current-toe-enthesis-feedback-10s-20261002/).

An offline prefix reconciliation against the matching 512 ms toe-prefix receipt
found identical device and runtime identities, all seven recorded input
SHA-256 values, and all 64 `human_standing_progress` rows for steps 8 through
512. The normalized newline-joined progress-row digest is
`10e40846981a7062b77dc90f011621ba369104be214b61ee908039f9c231327d`. This
confirms the reported first half-second is consistent across the two runs; it
does not establish bitwise q/v replay because these rows omit joint state,
per-contact wrench, and route-level force. The first recorded 10 mm drift at
1.576 s therefore occurs after this matching prefix, while its mechanical or
controller cause remains unresolved.
