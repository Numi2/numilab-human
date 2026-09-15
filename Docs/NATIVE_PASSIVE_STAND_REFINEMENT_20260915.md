# Native passive-stand refinement — 2026-09-15

The common-duration requalification is recorded in
[`Docs/media/native-passive-stand-refinement-20260915/receipt-v1.json`](media/native-passive-stand-refinement-20260915/receipt-v1.json).
It uses the same pose-24 source owner, passive-joint-tissue coupling, support
payload, tendon payload and NHEQ1 equality payload at each clock on the
physical Mac mini M4 Pro. The four runs cover one common `6.4 ms` duration:

| Clock | Steps | Peak acceleration | Static residual RMS | Penetration | Replay |
|---:|---:|---:|---:|---:|---|
| 100 µs | 64 | 0.642376363277 m/s² | 6.42342632457e-6 | 0 m | bitwise |
| 50 µs | 128 | 0.682456016541 m/s² | 6.42342632457e-6 | 0 m | bitwise |
| 25 µs | 256 | 0.918696343899 m/s² | 6.42342632457e-6 | 0 m | bitwise |
| 12.5 µs | 512 | 4.85533761978 m/s² | 6.42342632457e-6 | 0 m | bitwise |

The static solve is unchanged across the clock choices, but the release peak
range is `6.55840017994` times the minimum, exceeding the `5%` refinement
tolerance. `force_convergence` therefore remains open. This result confirms
that timestep reduction is no longer the useful fix: the next mechanics work
must make stiffness, fibre/tendon state, passive coupling and constraints part
of one converged dynamic solve. The receipt does not promote standing,
recovery, walking, anatomical contact, activation calibration, blood transfer,
materials or subject calibration.
