# Current native 12.5 µs release requalification — 15 September 2026

The current published native branch (`c45fa9622f6c73b58febdc24a7115aecf3d7699f`)
was run on the physical Mac mini M4 Pro with the one-adult 157-body/416-route
source, `NHCNT1` support contacts, `NHEQ1` equalities, `NHTENDON3`, and one
12.5 µs persistent Metal step. The exact command inputs and raw output are
retained in `Docs/media/native-current-release-20260915/`.

The source patch makes `0.8` the default recruitment ceiling for the
persistent stand when `--muscle-activation` is omitted. The older `1691e321`
one-step receipt remains retained as the historical `0.5` negative control;
the current default is covered by `receipt-v4.json` and its longer exact-clock
replay by `receipt-v5.json`.

The source root wrench still closes to `1.50167204538e-6 N`, the source
constraint preload is present (`2186.6550293 N` maximum), fibre equilibrium
performs two iterations, tendon force and moment residuals are below
`6.2e-5 N` and `3.2e-6 N m`, and penetration is zero. Those are useful handoff
checks, but they do not close the articulated state.

The historical `0.5` release fails the standing gate immediately. The compiled support state has
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

## Activation-1.0 sensitivity result

The same source was then rerun with `--muscle-activation 1.0` for one step and
64 steps, with an explicit `0.8` ceiling for 64 steps, and with the new
implicit default ceiling for 64 and 512 steps. The raw logs and immutable
`receipt-v2.json`/`receipt-v3.json`/`receipt-v4.json`/`receipt-v5.json` are retained in the same
media directory. This isolates recruitment sensitivity from the historical
activation-0.5 failure above:

| run | steps | horizon | compiled residual RMS | peak acceleration | penetration | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| activation 0.5 | 1 | 12.5 µs | 34.0593514806 | 196.14956665 m/s² | 0 m | failed |
| activation 1.0 | 1 | 12.5 µs | 0.0250606490784 | 0.486476838589 m/s² | 0 m | bounded release |
| activation 1.0 | 64 | 0.8 ms | 0.0250606490784 | 0.489560902119 m/s² | 0 m | bounded release |
| activation 0.8 | 64 | 0.8 ms | 0.0250606500378 | 0.491709738970 m/s² | 0 m | bounded release |
| omitted, source default 0.8 | 64 | 0.8 ms | 0.0250606500378 | 0.491709738970 m/s² | 0 m | bounded release |
| omitted, source default 0.8 | 512 | 6.4 ms | 0.0250606500378 | 32.7379798889 m/s² | 0 m | temporal drift |

The activation-1.0 run has six active supports of ten witnesses, no root
assistance, zero penetration, and `compiled_stand_balanced=true`. It is still
not a standing qualification: 64 base-clock steps cover only 0.8 ms, and the
scalar is a recruitment ceiling rather than a measured activation calibration;
the native static solve emits the per-route recruitment vector. The 0.8
ceiling gives the same bounded result within the recorded numerical spread,
which narrows the immediate failure to the default ceiling/recruitment and
coupled-equilibrium path without making the biological or sustained-standing
claim that the gate requires.

The 512-step replay completed on the physical M4 Pro with the same source and
implicit `0.8` ceiling. It retained six active contacts, zero penetration, no
root assistance, a `1.7372478851e-6 N` root residual and
`compiled_stand_balanced=true`, but the peak acceleration increased to
`32.7379798889 m/s²`; the maximum velocity and configuration deltas increased
to `0.000557818682864` and `2.52364020525e-6`. This is the current-branch
longer-horizon temporal-drift receipt (`receipt-v5.json`), not evidence of a
stable stand. The 6.4 ms horizon remains far below the sustained-standing,
recovery and walking gates.
