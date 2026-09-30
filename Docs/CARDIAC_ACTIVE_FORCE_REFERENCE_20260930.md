# Full-source ventricular active-stress internal residual - 30 September 2026

The pinned Rodero case18 candidate activation field now drives a **fixed-geometry
active-fibre internal residual** on all **1,097,534 ventricular tetrahedra** and
**218,077 unique source mechanical nodes**. The native Apple CPU producer and an
independent full-source NumPy assembly agree at two time samples; the latter
checks every tetrahedron and output node, global force and torque balance, and
the finite-element virtual-work identity. The output is a source-parameter
handoff for native electromechanics, not an accepted Matter step or a heartbeat.

The [source model configuration](../config/cardiac-wall-rodero18.v1.json)
pins the S4 phenomenological Tanh Stress values: 120 kPa peak isometric tension,
20 ms electromechanical delay, 50 ms contraction and relaxation constants, and
550 ms transient duration. The waveform follows the
[openCARP Tanh Stress equation](https://opencarp.org/documentation/examples/01_ep_single_cell/06_em_coupling).
The case18 source does not supply its length-dependence settings or a stress-free
reference state, so this candidate explicitly sets length dependence off and
evaluates the first Piola active stress at `F=I` on the CT mesh. Four symmetric
tetrahedral quadrature points interpolate nodal activation time. Supplied cell
fibre directions are normalized for evaluation; source bytes remain unchanged.

| Fixed-reference frame | 100 ms | 250 ms |
| --- | ---: | ---: |
| Active ventricular cells | 1,097,534 | 1,097,534 |
| Maximum cell tension | 101.919 kPa | 119.950 kPa |
| Stress-volume integral (`Pa m3`) | 10.659 J | 15.908 J |
| Sum of nodal internal-residual magnitudes | 1,218.04 N | 1,990.46 N |
| Maximum nodal internal-residual magnitude | 0.1061 N | 0.1607 N |
| Net internal-residual norm | 4.11e-14 N | 8.47e-14 N |
| Net internal-torque norm | 1.26e-14 N m | 7.27e-15 N m |
| Nodal affine virtual work | 0.011342866 J | 0.014825123 J |
| Analytic-vs-nodal virtual-work difference | 1.73e-18 J | -1.73e-18 J |
| Largest native-vs-independent force component difference | 6.94e-17 N | 7.81e-17 N |

The virtual displacement uses `H=diag(0.01,-0.004,-0.006)` solely as an
algebraic check. The stress-volume integral has energy units but is **not**
contractile work or an energy-closure claim. At 0 and 700 ms the same native
program produces all-zero residual buffers. Both nonzero frames replay bitwise;
a one-byte-mutated output fails the independent hash gate. The signed output is
`R_i = integral P grad(N_i) dV`, the **positive internal residual**. An external
nodal load would use `-R_i` and would require an accepted native owner.

Source assets, candidate activation, source supplement, active parameters,
producer source and binary, the two residual buffers, the independent gate,
zero controls and tamper control are hash-bound in
[`Docs/media/cardiac-active-force-reference-20260930`](media/cardiac-active-force-reference-20260930/).
The producer and verifier can be run with the pinned imported source asset:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.cardiac_active_force_reference \
  --asset Build/cardiac-electrical-source-20260930/asset \
  --activation Docs/media/cardiac-source-activation-20260930 \
  --output Build/cardiac-active-force-reference-20260930
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.cardiac_active_force_gate \
  --asset Build/cardiac-electrical-source-20260930/asset \
  --activation Docs/media/cardiac-source-activation-20260930 \
  --candidate Build/cardiac-active-force-reference-20260930 \
  --output Build/cardiac-active-force-reference-20260930/independent-gate.json
```

The candidate activation field still differs from the source CARP output, and
the three point-only LV/RV shared source IDs remain electrically split but
mechanically common in this reference mesh. There is no native accepted
electromechanical step, rollback history, deformation, preload equilibrium,
chamber pressure/flow response, ECG, or physiological heartbeat. In particular,
global zero force/torque and a virtual-work identity verify assembly arithmetic;
they do not qualify physiological motion or source-model reproduction.
