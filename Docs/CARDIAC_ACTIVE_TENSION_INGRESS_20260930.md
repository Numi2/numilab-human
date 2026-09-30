# Source-ordered ventricular tension and native FEM ingress — 30 September 2026

The pinned Rodero case18 candidate activation now produces a Float32 active
Cauchy-tension field for every one of the **1,470,083 source tetrahedra** at
100 and 250 ms. The **1,097,534 ventricular cells** carry candidate tension;
all other source labels carry zero. The two buffers retain source-tetrahedron
order and contain 5,880,332 bytes each. Maximum cell tensions are
101,918.9765625 Pa and 119,950.1328125 Pa respectively, within the source
120 kPa peak. These values follow the source-parameter Tanh Stress candidate
and the four-point quadrature defined in the
[fixed-geometry force reference](CARDIAC_ACTIVE_FORCE_REFERENCE_20260930.md).

An independent full-source gate assembled both Float32 fields into the FEM
internal residual on the source geometry. Against the earlier Float64
reference, the largest component difference was **3.95e-9 N** at 100 ms and
**4.62e-9 N** at 250 ms; relative L2 differences were 1.05e-7 and 6.83e-8.
The gate checks every source tension, the ventricular-only mask, source
geometry, positive cell volumes, and the complete 218,077-node residual.

[Numi Lab Matter `coupled` revision `6a0a483`](https://github.com/Numi2/numi-lab/commit/6a0a483222556b2675f66506a11123f15998386d)
adds an optional borrowed per-cooked-tetrahedron tension field to the owning
native FEM stress and tangent at ABI 35. It preserves the passive Guccione
law. On the local Apple M4, the focused source-material, material-frame and
active-tension runtime tests pass 3/3. The runtime probe accepts and bitwise
replays one source-cell active step; a zero field matches the omitted field,
and invalid tension values roll back. Wrong count and undersized buffers are
rejected before encoding.

The source-cell run uses **Rodero case18 cell 1**, its exact four source-node
positions, the source fibre/sheet-derived material frame, and the 100-ms
Float32 tension **89,809.78125 Pa**. Its free node moves
**1.39437375e-7 m** relative to the zero-tension control. The probe supplies
an explicit synthetic inertial density of **1,050 kg/m³**, fixes three nodes,
and uses a 1 µs timestep. This is a bounded native mechanics test, not an
anatomical boundary condition or a physiological motion claim. The native
source-cell receipt records the source, gate, code revision, binary hash,
input geometry, and two identical outer runs.

The published evidence is under
[`Docs/media/cardiac-active-tension-ingress-20260930`](media/cardiac-active-tension-ingress-20260930/).
Reproduce the source field and independent gate with the pinned imported asset:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.cardiac_active_tension_ingress \
  --asset Build/cardiac-electrical-source-20260930/asset \
  --activation Docs/media/cardiac-source-activation-20260930 \
  --output Build/cardiac-active-tension-ingress-20260930
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.cardiac_active_tension_gate \
  --asset Build/cardiac-electrical-source-20260930/asset \
  --activation Docs/media/cardiac-source-activation-20260930 \
  --candidate Build/cardiac-active-tension-ingress-20260930 \
  --output Build/cardiac-active-tension-ingress-20260930/independent-gate.json
```

The source-ordered field still needs an explicit source-to-cooked tetrahedron
mapping before any whole-wall encode. The case18 wall has **zero accepted
native electromechanical steps**. The source activation timing discrepancy,
stress-free reference, inertial densities, supports, chamber loads, and
measured motion remain unresolved. No blood-flow, heartbeat, or physical
anatomical qualification follows from the one-cell result.
