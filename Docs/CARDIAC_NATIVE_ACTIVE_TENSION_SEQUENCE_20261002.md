# Native ventricular active-tension sequence — 2 October 2026

The source-derived tension path now reaches one persistent native Matter FEM
runtime for eight consecutive 1 µs steps. Inputs sample the case18 ventricular
active-tension field at 100.000 through 100.007 ms. All eight full-source
fields passed the independent residual gate, then passed C++ source-to-cooked
Float32 mapping checks over 1,097,534 ventricular tetrahedra per frame.

On Apple M4, the active run accepted all eight native steps with status 0. A
fresh replay from the same package and inputs produced a byte-identical
13,957,120-byte state (`aaaf7174…856bf7d2`). The matched zero-tension control
also accepted all eight steps. It moved 60 nodes by at most `8.19e-12 m`; the
source-tension run moved 218,001 nodes by at most `8.06e-6 m`. The three
point-only LV/RV contact pairs remained at zero separation in the control and
reached `0.68`, `3.79`, and `1.01 µm` separation under source tension. The
three synthetic fixed nodes remained at their source positions, and active
and zero runs retained bitwise-identical nodal mass fields.

The independent field gate reports a maximum component difference of
`4.36e-9 N`, maximum relative L2 difference of `1.06e-7`, and maximum net
internal residual of `2.64e-13 N`. This checks source-field assembly and
residual parity. It does not establish a physiological heartbeat.

This bounded fixture uses synthetic `1050 kg/m³` density, zero gravity, and
three fixed nodes. It has no anatomical support, chamber pressure or flow,
valves, native electrical-to-mechanical coupling, calibrated material, or
subject reference. The source activation still differs from published CARP
timing. The accepted horizon is only `8 µs`, so it is a transient mechanics
subgate, not a cardiac cycle, heart qualification, or whole-Human result.

The [retained receipt and inputs](media/cardiac-native-tension-sequence-20261002/receipt.json)
bind source fields, independent gate, Matter revision and binary hashes,
sequence output, active/replay/zero node states, and the three contact-pair
measurements. The evidence auditor verifies every retained hash, all eight
source fields, native statuses, exact replay, source-rest movement, point-pair
separations, and optional local package/binary identities:

```sh
.venv-mujoco312/bin/python tools/audit_cardiac_native_active_tension_sequence.py \
  --receipt Docs/media/cardiac-native-tension-sequence-20261002/receipt.json \
  --matter-repo /Users/home/numi-lab-cardiac-native-20260930 \
  --package Build/cardiac-active-tension-native-sequence-20261002/native/ventricular.nmpkg \
  --asset Build/cardiac-electrical-source-20260930/asset
```

The existing [source-derived tension series](CARDIAC_ACTIVE_TENSION_SERIES_20261002.md)
documents field generation and the remaining activation-timing gap. The
Matter sequence runner is published at revision `7bcbf5d97e529abb425b81359d4457d7e91fe08f`.
