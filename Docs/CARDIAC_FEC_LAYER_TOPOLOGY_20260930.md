# Ventricular fast-layer topology sensitivity — 30 September 2026

The pinned Rodero case18 source does not provide its fast-endocardial
conduction (FEC) **cell tag**. The existing [source-parameter reconstruction](CARDIAC_SOURCE_ACTIVATION_20260930.md)
uses a declared *vertex one-ring* approximation: a ventricular tetrahedron is
fast if it touches any `UVC rho = 0` vertex and all four vertices lie at
`Z ≤ 0.7`. That marks 101,712 of 1,097,534 ventricular tetrahedra.

An [exact source-face audit](media/cardiac-fec-layer-topology-20260930/)
found that only **35,081** marked tetrahedra have a complete exterior
ventricular face with all three vertices at `rho = 0`. They own 35,082 such
faces because one cell has two. Of the other **66,631** vertex-ring cells,
32,792 touch the endocardium at one vertex, 33,837 at an edge, and two have
three or four `rho = 0` corners but no exterior endocardial face. The full
ventricular face quotient has 2,266,255 unique faces, 142,374 exterior faces,
and no face with more than two incident ventricular tetrahedra.

The audit retained a boundary-face-only FEC candidate and ran the same
anisotropic graph plus Apple CPU tetrahedral refiner with its own graph upper
bound. Two runs of that candidate replayed byte for byte. It changed the LV
activation span from **75.1892 to 75.2905 ms** and the LV 10–90% interval
from **37.3295 to 37.3967 ms**. Both move farther from the published case18
CARP simulation outputs, **71.9895 ms** and **29.8410 ms**, respectively.
Narrowing the geometric FEC layer therefore does not explain the measured
source-model mismatch in this reconstruction.

The [receipt](media/cardiac-fec-layer-topology-20260930/receipt.json),
[full-source mask](media/cardiac-fec-layer-topology-20260930/boundary-face-fec-candidate.u8le),
[arrival field](media/cardiac-fec-layer-topology-20260930/boundary-face-arrival.f64le),
and [auditor](../tools/audit_cardiac_fec_layer_topology.py) bind the source
asset, prior activation gate, exact face classification, compiled masked
refiner and replay. A modified source cell tag was rejected. Reproduce the
candidate with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  tools/audit_cardiac_fec_layer_topology.py --execute
```

This sensitivity test does **not** identify the missing CARP cell tag or
reproduce CARP's reaction-eikonal discretization. Neither geometric mask is
asserted as the published tag. The source comparator is simulation output,
not a patient measurement. Voltage, ionic state, atrial/AV conduction,
accepted native electrical steps, electromechanical feedback, and heartbeat
remain unqualified.

## Face-centroid interpretation — 2 October 2026

The [S4 supplement](media/cardiac-source-activation-20260930/rodero-s4-source.pdf)
describes the FEC layer as one element thick and limits its apicobasal extent
to `Z ≤ 0.7`. The earlier exact-face sensitivity required
all four tetrahedron vertices to satisfy the Z limit. A second discretization
now selects a ventricular tetrahedron when it owns an exterior triangular
face whose three vertices have `rho = 0` and whose mean face Z is at or below
0.7. This gives **35,454** tetrahedra from **35,455** endocardial faces.

The face-centroid field reaches an LV span of **74.8949 ms** and a 10–90%
interval of **36.8280 ms**. Against the published simulation outputs of
71.9895 ms and 29.8410 ms, the errors remain **+2.9054 ms** and **+6.9870 ms**.
Relative to the vertex-ring reconstruction, this discretization reduces those
errors by only **0.2943 ms** and **0.5015 ms**. Its largest local edge
travel-time violation is `1.55e-14 s`, and two native CPU runs replayed
bitwise. This small change does not resolve the timing gap.

The [face-centroid receipt and fields](media/cardiac-fec-face-centroid-20261002/receipt.json)
are reproduced by the same [auditor](../tools/audit_cardiac_fec_layer_topology.py)
with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m tools.audit_cardiac_fec_layer_topology --face-centroid --execute
```

This remains a geometrically motivated sensitivity candidate. The exact
case18 FEC cell tag is still absent, so the existing source reconstruction is
unchanged and neither candidate is identified as CARP's mask.
