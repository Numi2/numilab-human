# Native activation-cap requalification — 15 September 2026

The native Human probe now accepts `--whole-body-activation-cap` for the
source-bound static support certificate. The option is an explicit bounded
recruitment sensitivity control; it is not a calibrated physiological limit.
The change is on native source commit `fab66d5a8eefa18764cfd8b11c7220c60b034c38`
and binary SHA-256
`d9122b26b123e9ff407f32187404b1734f99ad78e1a7c1e12d00b632b58a2e2f`.

The physical M4 Pro runs use the same 157-body/416-route source, `NHCNT1`
support, `NHEQ1` equalities, 100 µs solver step, upper passive law, and full
128-row residual dump. The immutable outputs are in
`Docs/media/native-activation-cap-20260915/`.

| activation cap | sweeps | internal residual RMS | max root force residual | replay |
| ---: | ---: | ---: | ---: | --- |
| 1.0 (default) | 128 | `0.141417023854` | `1.86048202977e-6 N` | bitwise |
| 0.8 | 256 | `0.141417024066` | `1.72857460257e-6 N` | bitwise |
| 0.5 | 960 | `11.8273303596` | `1.25181838939e-6 N` | bitwise |

The default source recruitment remains the best of these runs, while the 0.8
cap is numerically indistinguishable and the 0.5 cap is infeasible for the
current pose/contact manifold. This rejects the idea that a lower global cap
alone fixes the standing failure. Every run still reports
`internal_balanced=false`; root support balance and replay pass, but dynamic
release, sustained standing, recovery, walking, activation calibration, and
subject calibration remain open.
