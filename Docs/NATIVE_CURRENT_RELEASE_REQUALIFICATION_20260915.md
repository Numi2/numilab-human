# Current native 12.5 µs release requalification — 15 September 2026

The current published native branch (`1691e321071001aa88e831d0f68e4314f12dd595`)
was run on the physical Mac mini M4 Pro with the one-adult 157-body/416-route
source, `NHCNT1` support contacts, `NHEQ1` equalities, `NHTENDON3`, and one
12.5 µs persistent Metal step. The exact command inputs and raw output are
retained in `Docs/media/native-current-release-20260915/`.

The source root wrench still closes to `1.50167204538e-6 N`, the source
constraint preload is present (`2186.6550293 N` maximum), fibre equilibrium
performs two iterations, tendon force and moment residuals are below
`6.2e-5 N` and `3.2e-6 N m`, and penetration is zero. Those are useful handoff
checks, but they do not close the articulated state.

The release fails the standing gate immediately. The compiled support state has
internal normalized residual RMS `34.0593514806`, maximum compiled acceleration
residual `164.118128055 m/s²`, and `compiled_stand_balanced=false`; the first
12.5 µs Metal step peaks at `196.14956665 m/s²`. The run uses seven active
supports out of ten and a global activation value of `0.5`, so it is also a
direct negative control against interpreting activation 1.0 as the sole cause.

This receipt is a retained **failed** exact-clock release. It proves source
identity, root support balance, preload transport, fibre initialization,
zero penetration, and native device execution for the attempted step. It does
not prove complete generalized equilibrium, sustained standing, recovery,
walking, anatomical loading, calibrated material or activation data, blood to
tissue transfer, or subject calibration.
