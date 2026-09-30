# Native ventricular initial-force parity — 30 September 2026

The corrected [source force reference](CARDIAC_SPLIT_ACTIVE_FORCE_REFERENCE_20260930.md)
now has a direct native Metal comparison. An opt-in Matter diagnostic copies
the first Newton assembly's element forces before solver scratch is reused.
The native 100 ms active run and a zero-tension control use the same
1,097,534-cell, 218,080-node [corrected ventricular package](CARDIAC_VENTRICULAR_POINT_CONTACT_SPLIT_20260930.md).
Subtracting zero from active isolates the constitutive tension response at
the initial candidate; the auditor converts native internal-force sign to the
positive source weak-form residual and assembles every cooked node.

The native residual differs from the independent Float64 source reference by
at most **4.61 × 10⁻⁷ N per component** and **5.37 × 10⁻⁶ relative L2**.
The native element-force capture and accepted node state replay byte for byte.
The active and zero accepted node hashes are unchanged from the prior
no-capture fixture, so the diagnostic did not change its physical outcome.
The three split LV/RV point contacts each pass the same full-field tolerance.
The observed error is within the gate's declared Float32 native versus Float64
source limits (`1e-6 N`, `1e-5` relative L2); it is not exact arithmetic parity.

The [receipt and assembled field](media/cardiac-runtime-force-parity-20260930/)
and [auditor](../tools/audit_cardiac_runtime_force_parity.py) bind the package,
source tension, source reference, native shader/runtime/tool
and metallib hashes. The three 67 MB native element captures and accepted node
buffers remain in the ignored local `Build/cardiac-runtime-force-parity-20260930`
directory; `--execute` recomputes all three and checks their retained hashes.
The auditor first reruns the independent full-source force gate, then compares
all 218,080 nodal vectors and the separately owned point-contact forces.

This is **one 1 µs synthetic-support mechanical fixture at the supplied CT
geometry**. The force is sampled before a native accepted update, while the
same active and zero transactions each accept a bounded step. There is no
native electrical evolution feeding tension, anatomical support, chamber
pressure/flow, recovered unloaded configuration, measured heart motion or
qualified heartbeat. The source activation reconstruction still differs
from the published CARP timing.

Recheck and reproduce on the local Apple host with:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  tools/audit_cardiac_runtime_force_parity.py --execute
```
