# Native Human force parity — 2026-09-13

The native M4 probe now checks the generalized muscle force that the static Human stand compiler admits against the force produced by the Metal MyoSim path before the persistent horizon advances. It requires equal full-vector dimensions and bounds the maximum absolute degree-of-freedom difference by `max(0.05 N, 1e-4 * max_abs_compiled_generalized_force)`; failure is terminal.

On `macmini` at native source commit `7c918e753f656cf42c6f9cab1640a7c7869453db`, the source-bound NHTENDON3 full-body probe used the exact `0.0000125 s` clock, one step, activation `1.0`, the authored six-contact stance, persistent Metal stand, and bitwise deterministic replay. It ran on Apple M4 Pro with all 416 source muscles. The measured source CPU/Metal generalized-force difference was `0.00644019908254 N`, and the gate passed. The static support result was balanced, with `1.86146132819e-06 N` maximum root-force residual and `0.0474679846966` maximum acceleration residual.

The audit also passed the four focused native vascular tests (`4/4`). The one-step dynamic trace itself remains only a diagnostic boundary: `persistent_max_acceleration=233.28062439`, `muscle_step_max_velocity_delta=0.00291600776836`, and `muscle_step_max_configuration_delta=3.64500962746e-08`. This closes source-force transfer parity, not sustained standing, recovery, walking, anatomy-to-mechanics coupling, blood mass transfer, material calibration, or subject calibration. The prior 512-step exact-clock trace with the same authored stance remains the controlling dynamic evidence and is still unstable.

Evidence and hashes are in [`Docs/media/native-force-parity-20260913`](media/native-force-parity-20260913):

- `identity.json` records the source and binary revisions, command boundary, metrics, and artifact hashes.
- `stdout.txt`, `stderr.txt`, `ctest.log`, `front.png`, `receipt.mrvpack`, and `visual.json` are copied from the native run.
- Source SHA-256: `34118b0d01df6d5f5f1bac306d84d71ab19eeb1103b54c9d7cf77c153c1e3b73`.
- Binary SHA-256: `6ee1257f295733d74bdfb810845729b3e420351ede9896eff960d2ec588b71dc`.
