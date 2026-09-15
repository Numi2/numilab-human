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

## Source-pinned fibre-state re-evaluation

The table above is a historical pre-repair result and must not be used as the
current dynamic release result. The later native runtime is published in
[`Numi2/numi-lab` branch `numi-human-passive-stand-20260915`](https://github.com/Numi2/numi-lab/tree/numi-human-passive-stand-20260915),
pinned at `e4a7494c0d9c914f10698dd1e87cb63b95cbb438`. A fresh build of that
source carried all 416 accepted static fibre lengths into the persistent
runtime, preserves a stationary compliant-fibre root at exactly zero path
velocity, and records a one-step-at-a-time production trace without changing
the mechanics.

The same `6.4 ms` refinement after that repair recorded the following native
status values. All four runs completed their requested steps, had zero
penetration, and replayed bitwise.

| Clock | Steps | Reported pre-projection peak acceleration | Tendon maximum force residual | Equality absolute-impulse sum |
|---:|---:|---:|---:|---:|
| 100 µs | 64 | 0.111788250506 m/s² | 6.29135975032e-5 N | 1.75639986992e-4 |
| 50 µs | 128 | 0.119950927794 m/s² | 3.49622787326e-5 N | 1.85146098374e-4 |
| 25 µs | 256 | 0.615386009216 m/s² | 8.63167442731e-5 N | 3.26470733853e-4 |
| 12.5 µs | 512 | 4.82679176331 m/s² | 8.63167442731e-5 N | 7.72664614487e-4 |

The `12.5 µs` peak is now localized rather than interpreted as a generic
solver failure: at step 496 it is the **pre-projection** acceleration of
equality row 50's dependent velocity coordinate (DOF 127). The row couples
dependent q/v `(128, 127)` to master q/v `(120, 119)`; after its analytic
velocity projection the coordinate is `7.24554411136e-7 m/s`, while the
largest published-state acceleration at that same step is
`0.0907450157683 m/s²` at DOF 77. The segmented trace endpoint agrees
bitwise with the unsegmented production horizon. This distinguishes an
unconstrained candidate correction from the final constrained trajectory; it
does **not** by itself establish force convergence or physiologic standing.

The trace now independently evaluates every equality-owned peak state in FP64
against the published equality polynomial. Across 358 such samples, the
largest position residual is `4.64732169414e-10` and the largest velocity
residual is `5.78374039776e-14 m/s`. At the 4.82679 m/s² step specifically,
the derivative is `0.0247611515387`, the position residual is
`4.49739146965e-10`, and the velocity residual is
`1.87239068210e-14 m/s`. This validates the final equality projection at the
observed states; it does not prove that the unconstrained candidate solve or
the full constrained trajectory have converged across clocks.

At the 12.5 µs common-duration run, continuous generalized virtual work was
`9.36397474459e-6 J` from source muscle force,
`-4.68560253300e-6 J` from static preload, and
`1.45731750358e-6 J` from support. These quantities deliberately exclude
impulsive contact and equality projection. The next acceptance comparison must
therefore retain trajectories, tendon residuals, equality impulses, and this
separate work accounting across clocks. No passive stiffness is tuned here to
obtain a standing pass, and `force_convergence` remains open.
