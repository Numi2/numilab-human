# Current source-bound toe enthesis requalification — 2 October 2026

The current MyoSim body manifest and 185-member BodyParts3D bone pack place
the four exact lesser-toe distal phalanges `42.807–45.375 mm` from the single
authored EDL/FDL route point. The earlier v5 result (`33.759–34.242 mm`) used
an older 184-member bone pack and its paired body manifest; it remains valid
for those older inputs, not the current compiled pair.

The compiler's toe-only semantic span bound is now `50 mm`. This applies only
to the source-declared map from each single lumped EDL/FDL endpoint to the four
named distal phalanges of digits 2–5. The route point and one source force law
remain unchanged, endpoint migration is zero, and the map does not create
independent toe actuators. Generic surface distance and patch bounds remain
`12 mm`; force amplification remains `4.0`.

The current rebuild admits all four EDL/FDL maps. Their maximum spans are
`42.807 mm` (right EDL), `43.734 mm` (right FDL), `44.317 mm` (left EDL), and
`45.375 mm` (left FDL). Closest-member distances from the source points range
from `3.599` to `7.365 mm`. The wrench maps preserve force to below
`6.5e-16` residual and moment to below `2.5e-17 m`; sampled total-force
amplification ranges from `1.788` to `2.749`.

Recompiling the current 832-endpoint pack also recovers seven previously
conditioning-limited endpoints through connected-face exact-surface patches,
under the existing distance, patch, and amplification limits. The candidate
contains 653 distributed envelopes and 179 explicit source-point fallbacks.
Those remaining point fallbacks are 24 endpoints on bodies without registered
bone surfaces, 153 endpoints beyond the 12 mm surface-distance gate, and two
source-authored non-bone endpoints.

The [retained candidate and one-step receipt](media/toe-enthesis-requalification-20261002/receipt-v1.json)
bind the NHTENDON3 payload, gzip-preserved compiler manifest (with its
decompressed content hash), full native stdout, stderr, Apple M4 binary, and
exact input hashes. The native package accepted one 1 ms Metal step with all
832 endpoint records and the new 653-envelope/179-point split. The two focused
importer regressions pass. This checks package admission and a bounded transfer
step only; the run reports
`compiled_stand_balanced=false` and does not qualify toe tracking, loaded foot
contact, standing, gait, clinical anatomy, or an integrated whole-Human result.
