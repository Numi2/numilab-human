# Human force and temporal convergence receipt

The published native activation owner `aecbdcf09f962db149709ee602142d73128335b6`
was exercised on the same prepared three-tiny-pelvis fixture at `100`, `50`,
and `25 us`. All three Apple M4 Pro trajectories pass the bounded execution
and bitwise replay checks over `1.6 ms`, with 16, 32, and 64 roots per scenario.
The fixture is not anatomical tissue, loaded standing, walking, or calibration
evidence.

The common-grid reducer retains both scenarios and all applied-force samples:

| comparison | maximum activation difference | maximum applied-force difference |
| --- | ---: | ---: |
| 100 vs 50 us | `6.3240528e-05` | `0.3154526 N` |
| 50 vs 25 us | `3.0636787e-05` | `0.3554077 N` |

Activation error halves as expected for the exact first-order hold, but the
source applied-force difference increases. The force-convergence acceptance
therefore remains **failed/partial**. The measured next owner is coupled
state/force integration (the q/v maxima likewise increase from `0.0194913` to
`0.0335683`), not the activation update.

Evidence: [refinement observations](media/activation-exact-20260913/native/refinement-observations.json),
[100 us trace](media/activation-exact-20260913/native/prepared-trajectory-100us.log.gz),
[50 us trace](media/activation-exact-20260913/native/prepared-trajectory-50us.log.gz),
and [25 us trace](media/activation-exact-20260913/native/prepared-trajectory-25us.log.gz).

This receipt does not qualify the exact `12.5 us` clock beyond its separate
admission receipt, and does not close anatomical supports/loading, held-out
activation calibration, two-way blood mass/momentum transfer, unresolved
materials, sustained standing/walking, or single-male release.

The separate [exact-clock persistent-stand refinement](EXACT_STAND_REFINEMENT_20260913.md)
uses a common `0.8 ms` duration and the authored six-contact stance at
`100/50/25/12.5 us`. Static support-wrench residual is stable at
`1.87e-6 N` and all four runs replay bitwise, but terminal acceleration remains
about `8.67e3 m/s^2` with `compiled_stand_balanced=false`. It therefore
localizes the remaining convergence problem to loaded dynamic
joint/fibre/contact state and does not promote the force or behavior gates.
