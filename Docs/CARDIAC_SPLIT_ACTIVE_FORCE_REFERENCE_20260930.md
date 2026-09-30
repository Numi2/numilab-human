# Active-force reference for the corrected ventricular topology — 30 September 2026

The prior [full-source active-force reference](CARDIAC_ACTIVE_FORCE_REFERENCE_20260930.md)
assembled its fixed-geometry internal residual on 218,077 unique source nodes.
The [native ventricular cook](CARDIAC_VENTRICULAR_POINT_CONTACT_SPLIT_20260930.md)
now has 218,080 mechanical nodes because three LV/RV source nodes meet only at
a point. Continuing to sum both ventricles' force contributions into one node
would imply a mechanical path across those point contacts.

The Apple CPU producer transfers each point-only RV cell's active-fibre
contribution from the old shared residual into its own corrected RV node. It
remaps the prior residual to exact cooked-node order, changing only those three
former shared entries and adding the three RV entries. It
uses the source activation arrival, Rodero candidate Tanh Stress parameters,
fibre direction, and tetrahedral quadrature for the three transfers. At 100 ms
their force magnitudes are **0.000780, 0.006888, and 0.001181 N**; at 250 ms
they are **0.002055, 0.008111, and 0.002814 N**. The old pairwise sum remains
the same force, while the two sides now own separate states.

The [independent full-source gate and retained fields](media/cardiac-split-active-force-20260930/)
verify the previously published 1,097,534-cell reference first, rebuild each
of the three RV contributions from the pinned mesh, and compare all 218,080
corrected force vectors. The largest native-versus-independent component
difference is `2.61e-18 N`. Each 100/250 ms field and native receipt replays
byte-for-byte; net internal force, torque, and affine virtual work retain the
prior source balance. The gate binds the corrected source-to-cooked node map,
producer source and binary, source activation, and prior force evidence.

This is an **offline internal-force reference at the supplied CT geometry**.
The 100 ms source-ordered tension was separately accepted by the bounded
native FEM fixture, but this exact 218,080-vector field has not been compared
with the runtime's full-source assembled force. The source activation timing
still differs from published CARP output. Anatomical supports, unloaded heart
geometry, pressure/flow coupling, accepted electrical evolution, physiological
motion, and heartbeat remain unqualified.

Recheck the retained field and rerun the native producer with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.cardiac_split_active_force_gate \
  --output Docs/media/cardiac-split-active-force-20260930/independent-gate.json \
  --execute
```
