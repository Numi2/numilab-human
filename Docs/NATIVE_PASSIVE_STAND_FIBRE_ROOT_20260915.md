# Native passive-stand fibre-root repair, 15 September 2026

The persistent Human stand previously discarded the static solver's accepted
fibre lengths and initialized the Metal horizon from the zero-length sentinel.
The native repair on the physical Mac mini carries all 416 accepted static
fibre lengths into the runtime state. It also preserves an already accepted
zero-velocity constitutive root when float-length bisection cannot resolve a
sub-micron correction at the canonical clock. The initial fibre equilibration
remains an explicit fixed-pose preparation step and is not counted as a
physical time advance.

The source branch is
`numi-human-passive-stand-20260915` at
`f45fcfdc80a05c2226d638227295add7f789c55c`; the physical M4 Pro binary is
`62648144c20f386b25f625d503fd101f5c846b9ca8b0dac0649e4a064fb04919`. The
immutable source patch is retained beside the native transcripts in
[`media/native-passive-stand-fibre-root-20260915`](media/native-passive-stand-fibre-root-20260915).

Both cases use the same source passive tissue, support and tendon/equality
payloads at `12.5 us`. The 64-step (`0.8 ms`) release peaks at
`0.115904301405 m/s2`; the canonical 512-step (`6.4 ms`) release peaks at
`4.82679176331 m/s2`. Static normalized residual RMS remains
`6.42342632457e-6`, penetration is zero, and deterministic replay is bitwise in
both cases. The repair therefore closes the initial fibre-state ownership
defect and bounds the short release, while the long release still fails the
force-convergence and standing gates. The next native owner is the coupled
long-horizon state/force solve; lowering the clock again is not evidence of a
solution.

The structured receipt is
[`receipt-v1.json`](media/native-passive-stand-fibre-root-20260915/receipt-v1.json),
with the immutable manifest in the same directory. It deliberately keeps
anatomical support loading, activation calibration, blood mass transfer,
material calibration, subject calibration, sustained standing, recovery and
walking false.

## Published runtime tuple

The retained historical binary receipt above remains historical evidence. The
actual native implementation is now pinned at the public Numi Lab tag
`human-native-step281-rank-audit-20260915` (`337741b51bfc4a5552a837dacb4d0b5c4268d298`),
and the five exact runtime inputs plus a fresh public-tag replay are pinned at
the companion Human source-input tag
`human-native-runtime-source-inputs-20260915`. See
[`NATIVE_RUNTIME_SOURCE_PUBLICATION_20260915.md`](NATIVE_RUNTIME_SOURCE_PUBLICATION_20260915.md)
for the complete tuple, hashes, reconstruction command, and its deliberately
limited qualification boundary.
