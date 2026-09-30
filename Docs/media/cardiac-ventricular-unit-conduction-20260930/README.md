# Ventricular unit-conduction diagnostic, 2026-09-30

The source-bound Apple CPU executable takes one explicit graph-diffusion step
over all **1,097,534** Rodero ventricular tetrahedra. Its 218,080 electrical
DOFs share the 2,628 nodes on verified complete LV/RV interface faces and keep
three point-only contacts split. With an initial **dimensionless** field of 1
on the LV and connected interface and 0 on the RV-only nodes, the connected
step changes 2,139 RV-only DOFs. Duplicating the complete-face interface in
the blocked control gives zero RV-only changes. An independently assembled
residual matches the native field to `1.11e-16` maximum absolute error; the
volume-weighted field integral is conserved within `1e-12`, the graph energy
decreases, and a second native execution is byte-identical.

The coefficient of 1 m²/s is a **synthetic fixture**. The 50.63 ns step is
only its stability-bound update interval, not a measured conduction time.
The graph energy is a mathematical diffusion functional, not Joules. This is
not a membrane-voltage or ionic model, ECG, atrioventricular or atrial
conduction, contraction, perfusion, or a heartbeat. It is a standalone native
CPU diagnostic: **zero HumanPack accepted electrical steps**. The prior
source-activation gate retains the source-model timing mismatch and does not
qualify the source simulator's electrical model.

The retained `connected*`, `connected-replay*`, and `blocked*` files are the
exact fields, DOF maps and native receipts. `independent-audit.json` binds
them to the source anatomy manifest, published source-node quotient, previous
complete-face topology gate, native source and auditor source. The binary was
compiled with Apple clang 21 on macOS 26.6, arm64; `execution.json` records its
hash and build details. The source anatomy asset is in
`Build/cardiac-electrical-source-20260930/asset`, and the source-node quotient
is in `Docs/media/cardiac-source-activation-20260930`. The pinned source import
and its known limitations are documented in
[`CARDIAC_WALL_ANATOMY_20260912.md`](../../CARDIAC_WALL_ANATOMY_20260912.md).

Rebuild and rerun from the repository root with the source asset present:

```sh
mkdir -p Build/cardiac-ventricular-unit-conduction-20260930
clang++ -std=c++20 -O3 -Wall -Wextra -Wpedantic -Werror \
  tools/cardiac_ventricular_unit_conduction.cpp \
  -o Build/cardiac-ventricular-unit-conduction-20260930/cardiac-ventricular-unit-conduction
```

The positional arguments are the asset directory, published source-node map,
published DOF-region map, output field and output map. For example:

```sh
Build/cardiac-ventricular-unit-conduction-20260930/cardiac-ventricular-unit-conduction \
  Build/cardiac-electrical-source-20260930/asset \
  Docs/media/cardiac-source-activation-20260930/ventricular-source-nodes.u32le \
  Docs/media/cardiac-source-activation-20260930/ventricular-dof-regions.u32le \
  Build/cardiac-ventricular-unit-conduction-20260930/connected.f64le \
  Build/cardiac-ventricular-unit-conduction-20260930/connected-nodes.u32le \
  > Build/cardiac-ventricular-unit-conduction-20260930/connected.json
```

Repeat for `connected-replay` with its output names; for `blocked`, use its
output names and append `--block-interface` before the output redirection.
Then verify the retained evidence:

```sh
PYTHONPATH=src:. .venv-mujoco312/bin/python \
  tools/audit_cardiac_ventricular_unit_conduction.py \
  --run Docs/media/cardiac-ventricular-unit-conduction-20260930 \
  --output Build/cardiac-ventricular-unit-conduction-20260930/recheck.json
```
