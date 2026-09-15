# Native coupled-velocity closure diagnostic — 2026-09-15

The isolated native branch `1fd04506` fixes a verifier boundary in the
experimental coupled-velocity-closure trace. That kernel candidate deliberately
retains the terminal dependent velocity after the coupled sweep, so the host
must require exact position projection while recording the velocity difference
as a diagnostic. The old verifier required both and rejected the candidate
after a successful device transaction.

The physical Mac mini M4 Pro replay uses the source `NHRIGID2`, `NHMYO2`,
`NHCNT2`, `NHTENDON3`, and `NHEQ1` inputs at the canonical 12.5 microsecond
clock for 64 accepted steps. It has six active source support contacts, zero
penetration, bitwise deterministic replay, static normalized residual RMS
`9.0656247312e-6`, and a bounded persistent peak acceleration of
`0.115922890604 m/s2`. The dynamic force audit peaks at `0.032620927143 N`.

The equality-dependent velocity error is retained (`3.09879681026e-7` in the
terminal summary); it is the expected consequence of the diagnostic flag and
is not silently counted as exact velocity closure. The receipt therefore
records a bounded candidate replay and the corrected verifier, while keeping
production force convergence, sustained standing, recovery, walking, anatomy,
activation calibration, blood/tissue mass transfer, materials, and subject
calibration open.

The 512-step extension was attempted with the same inputs and clock. The
native process remained in GPU wait for a bounded six-minute window, emitted
only payload-admission lines, and was terminated with `SIGINT` before a result
was published. The typed [timeout receipt](media/native-coupled-velocity-closure-diagnostic-20260915/long-horizon-timeout-v1.json)
keeps that failure visible and leaves long-horizon force convergence and
standing open.

The full native stdout/stderr and source patch are stored beside
`receipt-v1.json`. The native source checkout is isolated from the dirty
`/Users/n/MetalRobo-human` checkout.
