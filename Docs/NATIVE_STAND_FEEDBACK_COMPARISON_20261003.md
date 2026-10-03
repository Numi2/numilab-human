# Source-path feedback onset comparison — 2026-10-03

The registered 1.6-second comparison found that path feedback increased the
source stance's horizontal root drift by **7.973 mm**. The no-feedback control
drifted 2.927 mm; the feedback trajectory drifted 10.900 mm. This contradicts
the fixed prediction that feedback would reduce drift by 5–50 mm.

The comparison used the Apple M4 Pro native MyoSim owner, source-path feedback
gains 10 and 1 s, 64 contact iterations, a 1 ms step, 1,600 steps, and no root
assistance. Both arms used the exact same input/runtime hashes and matching
initial q/v hashes:

- Initial q SHA-256: `3115b8ffeb069867e95b938b418779a6ea9484731c39a6976b34993e3b4c5fb8`
- Initial v SHA-256: `48be08a3472ef9752fd789dc87f1a942b58da5d73c15e6d3c10f73dd4667bce1`

All 200 accepted progress samples in each arm retained six support contacts.
The largest reported penetration was 2.02 µm in control and 2.15 µm in
treatment. Root assistance stayed at zero. The feedback correction peaked at
0.1412, below its 0.2 bound. Treatment first exceeded 10 mm horizontal drift
at 1.576 s; control remained below 10 mm through 1.6 s. Treatment root speed
peaked at 0.0698 m/s before ending at 0.00505 m/s.

The trial data and immutable receipts are in
[`study-followup-v3`](media/native-stand-feedback-comparison-20261003/study-followup-v3/).
The registered analysis reports `contradicted`, and `numi science verify`
recomputes that result with no integrity errors. The registration hash is
`d168f9ec8c603dea8f3ff061214bb04815909242a8f8519b4453679df6839f77`.
The detailed row-wise values are in
[`trajectory-comparison.json`](media/native-stand-feedback-comparison-20261003/trajectory-comparison.json).
All 200 accepted progress rows from the wrapper-failed treatment attempt match
the clean treatment rerun exactly, confirming a deterministic replay of that
trajectory while leaving the first attempt's native exit code unknown.

There were two wrapper errors, both retained rather than hidden. The control
was launched outside `numi science run` by an incorrectly formed syntax-check
command, so its 200 raw progress rows and terminal state are preserved but its
native exit code is unknown. The first treatment reached step 1,600, then its
SSH/zsh wrapper assigned to zsh's read-only `status` parameter and failed
before capturing the native exit code. That trial remains invalid and its
notebook analysis is inconclusive. A corrected wrapper reads the exact script
from stdin, passed remote zsh syntax validation, and the clean treatment run
recorded native exit code 0. The original prediction was sealed before the
control; the follow-up plan retained the same model and prediction.

This is one source-specific simulation pair, not population inference. The
control exit-status gap limits the strength of the comparison even though its
endpoint, q/v identity, and input hashes match. Both runs use the retained 2
October runtime (`c35f0082ffcc29806d8bece2e31177b64fe7e5c0`) and input package;
they do not requalify current-main standing, the 3 October neutral-coordinate
overlay, clinical anatomy, or integrated physiology. The progress output
provides root-level accepted summaries rather than per-route force and full
per-step q/v. The next useful standing test needs a feedback-compatible trace
of per-step q/v and source-route forces around the 1.576-second onset. The
retained probe source currently rejects persistent trace capture when feedback
is enabled (`capturePersistentStandTrace` is excluded by the feedback gate),
so that instrumentation needs a separate feedback-safe telemetry path rather
than reusing the unassisted energy/replay trace gate.

Evidence locations:

- [Original preregistered plan and stopped notebook](media/native-stand-feedback-comparison-20261003/study/)
- [Out-of-band control streams and provenance](media/native-stand-feedback-comparison-20261003/out-of-band-control/)
- [Wrapper-failed treatment streams and provenance](media/native-stand-feedback-comparison-20261003/out-of-band-treatment-wrapper-failure/)
- [Corrected follow-up notebook and analysis](media/native-stand-feedback-comparison-20261003/study-followup-v3/)
- [Runtime source snapshot and patch](media/native-stand-feedback-comparison-20261003/source-snapshot/)
