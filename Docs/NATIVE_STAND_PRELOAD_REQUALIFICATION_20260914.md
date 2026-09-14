# Native Human source-constraint preload requalification — 2026-09-14

This receipt covers the isolated native source branch
`numi-human-equilibrium-20260914` at `ee5cb816e25de819f4dcf683287ac7cad59b0523`,
based on `f239c6314bc3912c641db6eb909525eaa098f20a`. The source tree was built
in `/private/tmp/numi-human-native-fiber-f239-build` and executed on the
physical Apple M4 Pro. The shared Mac mini production checkouts were not
modified.

The increment carries the accepted static joint-equality and position-limit
reaction, one `nv` vector per environment, into the first persistent Metal
release. The device stand arena reserves the prefix explicitly and adds it to
generalized muscle/contact effort. The contact/equality parity probe clears the
preload so its CPU/Metal comparison remains an independent check. This is a
source-equilibrium warm start; it is not a monolithic sparse KKT solve and does
not establish sustained standing.

## Inputs and build

| Input | SHA-256 |
| --- | --- |
| rigid NHRIGID2 | 6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44 |
| muscle NHMYO2 | 9a988f19a6fd8e5335fb0cf2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05 |
| support NHCNT2 | c7712daf79cd8a589a6d23942a4df84a7da928e5911455ce19078f9b24daaaf4 |
| equalities NHEQ1 | b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a |
| tendon NHTENDON3 | a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555 |
| final native binary | 288fc13eb196ea9479cc350d685dc1ead681f3c5e4edf1ba1cd34fc8a1aec912 |

The qualification used the one-adult male 157-body package, 416 source routes,
128 velocity DoF, 51 joint equalities, NHCNT2 supports, NHTENDON3 transfers and
the canonical `--muscle-step-seconds 0.0000125` clock.

## Static and preload ownership

The source certificate and fibre handoff remain unchanged and pass:

| Quantity | Result |
| --- | ---: |
| total support | 952.864475233 N |
| maximum root-force residual | 1.8052625137e-6 N |
| internal normalized residual RMS | 0.0250606490272 |
| active support contacts | 9 / 18 |
| internal balanced | true |
| fibre equilibration iterations | 2 |
| source/runtime support-force difference | 0 N |
| maximum preloaded reaction | 1363.89331055 N |

The final rebuilt one-step receipt reports
`source_constraint_preload=static_equality_plus_position_limit` and
`source_constraint_preload_max_n=1363.89331055`.

## Native release evidence

| Horizon | Peak acceleration | Maximum velocity change | Maximum configuration change | Penetration | Replay |
| --- | ---: | ---: | ---: | ---: | --- |
| first 12.5 µs release | 0.488360792398 m/s² | 1.21709117593e-5 | 2.27966409638e-10 | 0 m | not requested |
| 512 × 12.5 µs, no replay | 112.067703247 m/s² | 0.00101080874447 | 6.2084077399e-6 | 0 m | not requested |
| 512 × 12.5 µs, two passes | 112.067703247 m/s² | 0.00101080874447 | 6.2084077399e-6 | 0 m | bitwise |

The 512-step run reports `persistent_completed_steps=1024` because the
qualification executes the horizon twice for the replay check. Equality
impulse remains finite (`stand_max_equality_impulse=2.21788354793e-6`, total
`0.00371364830062`). These numbers show a large reduction from the prior
72be18e9 baseline of 214.411651611 m/s² at one step and 149823.15625 m/s² over
512 steps. They do not qualify standing: the horizon is only 6.4 ms, uses
activation 1.0, and has no 10 s/60 s assistance-free hold or perturbation
recovery.

## Completion boundary

This increment proves a reproducible source-reaction warm start and a bounded,
zero-penetration 512-step native release with deterministic replay. It leaves
the coupled sparse/matrix-free KKT solve, sustained standing, active-set contact
calibration, closed-loop activation recruitment, walking and the 420-trial gate
open. The anatomical supports/loading, blood-to-tissue mass transfer,
calibrated tissue/material data and subject calibration gates remain separate
and unresolved; source inventories and synthetic circulation owners are not
physical whole-human closure.

## Raw artifacts

- [one-step.log](media/native-human-constraint-preload-20260914/one-step.log) — SHA-256 `86e2f084e6d15e0467723f0322ffee0068133d810c2890eda4ea806b4d394461`
- [horizon-512.log](media/native-human-constraint-preload-20260914/horizon-512.log) — SHA-256 `aef78fc83443bb4d4124ce063f553e9498c277872879ac014854ea89e76d0984`
- [horizon-512-replay.log](media/native-human-constraint-preload-20260914/horizon-512-replay.log) — SHA-256 `b86af2963b7744a7c7669a08d8307109b2357838ecb83d1f47ee5d9b20edcc14`

The replay artifact was run from the same preload implementation before the
receipt-only output fields were added; the final rebuilt binary and applied
preload are independently captured by `one-step.log`.

## Retained failed diagnostic

A follow-up experiment also preloaded the published static generalized
residual. It held the 64-step horizon at `0.467938244343 m/s²`, but the
512-step transaction never reached a terminal receipt and remained blocked in
`MTLCommandBuffer waitUntilCompleted` after more than four minutes. The
isolated process was terminated; this correction is not in `ee5cb816` and is
not a standing result. The partial input/output log is retained at
[residual-preload-512-timeout.log](media/native-human-constraint-preload-20260914/failed-diagnostics/residual-preload-512-timeout.log)
with SHA-256 `fcb7d21415270d9744bec770baf41c850cf36be77f85851a774bd2c203d21b27`.

A separate one-step-only preload handoff was also rejected: after clearing the
preload at step two, the 64-step run reached `8755.6796875 m/s²`, a
`0.328599065542` velocity change, and `1.58839938535e-7 m` penetration. Its
partial log is retained at
[first-step-only-preload-64.log](media/native-human-constraint-preload-20260914/failed-diagnostics/first-step-only-preload-64.log)
with SHA-256 `74afe4ef48c053167046c4d3fad626774ec186e4489a7f31ce88fd4bf1c7d73e`.
