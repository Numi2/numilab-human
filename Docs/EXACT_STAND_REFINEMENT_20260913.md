# Exact-clock persistent standing refinement

The current native owner was run on the physical Apple M4 Pro using one
adult-source full-body fixture, the authored six-contact stance, all 416 source
muscle routes at activation `1.0`, NHTENDON3 transfer, joint equalities, and no
root assistance. The four runs cover the same `0.8 ms` duration at `100`, `50`,
`25`, and `12.5 us`; each run includes bitwise deterministic replay.

| step | roots | maximum acceleration | maximum penetration | terminal configuration drift | static root-force residual | balance |
|---:|---:|---:|---:|---:|---:|:---|
| 100 us | 8 | 8,662.21 m/s² | 10.90 µm | 3.169e-4 | 1.872e-6 N | false |
| 50 us | 16 | 8,669.50 m/s² | 4.65 µm | 2.275e-4 | 1.872e-6 N | false |
| 25 us | 32 | 8,670.61 m/s² | 1.17 µm | 2.219e-4 | 1.872e-6 N | false |
| 12.5 us | 64 | 8,672.60 m/s² | 0.294 µm | 2.190e-4 | 1.872e-6 N | false |

The exact clock, support-wrench closure, assistance-free execution, and replay
are reproducible engineering properties of this fixture. The terminal dynamic
acceleration remains about `8.67e3 m/s²`, the compiled residual reports
`balanced=false`, and the configuration drifts during the bounded horizon.
Consequently this result does not close force convergence, sustained standing,
recovery, walking, anatomical loading, activation calibration, material
calibration, or blood/tissue qualification. It identifies the next native
owner as the loaded dynamic joint/fibre/contact solve and its state history,
not the clock admission or static support-wrench reducer.

The complete native stdout/stderr logs and hash-bound receipt are in
[`Docs/media/exact-stand-refinement-20260913/`](media/exact-stand-refinement-20260913/).
The receipt binds native commit `cfef55a8199c2167fbe9beb04417a47db705db5d`,
the clean Mac mini worktree, the Apple M4 Pro binary, all four timestep runs,
and the source fixture. This is retained failed/partial evidence rather than a
behavior qualification claim.
