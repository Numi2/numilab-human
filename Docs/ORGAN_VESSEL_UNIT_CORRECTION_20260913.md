# Corrected vessel source/world moment units and mass-moment candidate, 13 September 2026

The previous vessel registration receipt applied the pinned
`global_source_mm_to_myosim_world_m` matrix directly to moments that had
already been converted from authored millimetres to metres by the organ-moment
compiler. That applied the millimetre factor twice. For example, the previous
FJ1932 world volume was `3.976317887938874e-14 m³` and its centroid z was
`0.12494428048183852 m`.

The registration compiler now applies `uniform_scale_after_mm_to_m` once to
the metre-valued source moments and adds the pinned world translation. The
corrected FJ1932 values are `3.976317887938873e-05 m³` and
`1.1858212301670572 m`. Central second volume moments are scaled by the fifth
power, as required for a volume second moment. The old receipt remains
immutable historical evidence; the corrected receipt is
[`registration.json`](media/organ-vessel-registration-corrected-20260913/registration.json)
with SHA-256
`002d0fe7dc077de726d3b2454735a5a7d3a9f8a26144888735de47c15239397e`.

Using that corrected source/world receipt, the new
[`vessel-mass-moment-owner-candidate`](media/vessel-mass-moment-owner-corrected-20260913/receipt.json)
computes exact zeroth, first, and second mass moments for all six registered
vessel surfaces with the explicit `1060 kg/m³` engineering candidate density.
Its receipt SHA-256 is
`2ac61a232c5ba2e91311428067ea6c57644f098a45290ef6570e85d62faa0c40`.
It records `0.00023107836345349678 m³` of authored source-surface volume and
`0.0002363949664537634 m³` after the pinned world scale,
`0.25057866444098925 kg` of proxy mass, six unique owners, zero initial
momentum, and an atomic checkpoint/restore digest. Its status is `partial`:
the density is not a subject measurement, the source surfaces are not admitted
lumen volumes, and no pressure gradient, tissue exchange, vessel-wall
mechanics, or anatomical blood-mass owner is promoted.

The corrected ownership ledger is retained at
[`ledger.json`](media/organ-blood-ownership-ledger-corrected-20260913/ledger.json)
and binds the corrected registration and mass-moment hashes while keeping
anatomical blood-mass transfer false. Its SHA-256 is
`bb49c0d4daad6b8dd6a376bba96a4907fb7ac7fc0142ef6b7c6e31537a8d0c1e`.
The physical Mac mini rerun against the corrected fixture passes the bounded
source/world checker; its retained output is
[`native-rerun.txt`](media/native-vessel-registration-corrected-20260913/native-rerun.txt).
That native result still does not admit tube/lumen mechanics, pressure transfer,
tissue exchange, calibrated density, or anatomical blood ownership.
