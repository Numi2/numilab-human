# Native stance placement and exact-clock horizon

The persistent one-male run is now repeated from the authored source stance
profile instead of the unplaced reference pose. The physical Apple M4 Pro fit
uses the root-height coordinate, the six ankle/subtalar/MTP coordinates, and
the six active foot witnesses from
[`config/myosim-support-stance.v1.json`](../config/myosim-support-stance.v1.json).
The support pose fit converges in two iterations with sub-picometre witness
gaps. Six of ten support witnesses carry load; the static compiled wrench is
`952.864475177 N` with `1.86146132819e-6 N` maximum floating-root residual,
and `compiled_stand_balanced=true`.

The same source state completes 512 exact `12.5 us` steps (`6.4 ms`) on the
physical M4 Pro and replays bitwise. The run includes the source support
payload, NHTENDON3 transfer, all 416 recruited routes, activation cap `1.0`,
and the hash-bound torso visual payload. It still exposes a dynamic failure:
peak acceleration is `108763.164 m/s²`, velocity change is `1.30699`, and
configuration change is `0.00444473`. These metrics retain force/temporal
convergence and sustained standing as open.

The native source revision is `de83ba9bbde08fee921c1a2ff1e567f981714326`.
The complete run, four frames, stdout/stderr, identity, receipt, and focused
`4/4` CTest log are retained in
[`Docs/media/native-stand-stance-20260913/`](media/native-stand-stance-20260913/).

This closes the authored source stance placement, static support-wrench
admission, and source activation/recruitment subgates for one adult male. It
does not qualify sustained standing, recovery, walking, calibrated activation,
organ or vessel mechanics, materials, density, blood/tissue exchange, or
subject calibration.
