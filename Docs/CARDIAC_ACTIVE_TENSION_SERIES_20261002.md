# Source-derived ventricular tension series — 2 October 2026

The active-tension ingress now accepts strictly increasing sample times and
produces a source-ordered Float32 field for all 1,470,083 case18 tetrahedra at
each requested time. Six samples from the existing ventricular activation
candidate cover 100, 150, 250, 400, 600 and 650 ms. The peak cell tension rises
from 101.9 kPa at 100 ms to 120.0 kPa at 400 ms, then falls to zero by 650 ms.
At 650 ms all 1,097,534 ventricular cells have zero tension and all other
tetrahedra remain zero throughout.

The independent full-source gate recomputes the tension and assembles its FEM
internal residual at every sample. Its largest component difference is
4.63e-9 N, its largest relative L2 difference is 1.06e-7, and the largest net
internal residual norm is 2.88e-13 N. At the pinned 100 and 250 ms samples, the
independent Python reconstruction matches the archived C++ residual within
7.81e-17 N. The zero-tension endpoint now passes with
zero residual and zero relative error; this fixes the gate's zero-denominator
edge case while retaining an exact-zero check. A negative control that changes
one source cell to 1 Pa, updates its declared frame hash and cell summary, is
rejected by full-source residual parity.
The default 100/250 ms profile also regenerates and passes the current gate.

The [source and gate receipt](media/cardiac-active-tension-series-20261002/receipt.json)
binds the six fields, source activation summary, anatomy manifest, producer and
independent gate. Reproduce it from the repository root with:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.cardiac_active_tension_ingress \
  --asset Build/cardiac-electrical-source-20260930/asset \
  --activation Docs/media/cardiac-source-activation-20260930 \
  --output Build/cardiac-active-tension-series-20261002 \
  --times-ms 100 150 250 400 600 650
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.cardiac_active_tension_gate \
  --asset Build/cardiac-electrical-source-20260930/asset \
  --activation Docs/media/cardiac-source-activation-20260930 \
  --candidate Build/cardiac-active-tension-series-20261002 \
  --output Build/cardiac-active-tension-series-20261002/independent-gate.json
```

This is a sampled single-transient tension candidate driven by the existing
fixed activation-arrival field. The activation reconstruction still differs
from published CARP timing. These samples do not represent repeated electrical
excitation or an accepted native anatomical step. The four-chamber wall still
has zero accepted electromechanical steps; heartbeat, chamber loading,
circulation, calibrated reference state and measured motion remain open.
