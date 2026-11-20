# Match shell geometry and mass to the native reference

Apple M6 on SSH Numi. This increment starts from native crossing evidence
at 0d05ab99 and the exact baseline Metal binary and replay inputs. The
native capture advanced one accepted 15.625 μs root with variable pleural
pressure enabled. Its initial Float32 shell and the previous exact
crossing audit match bytewise. The capture exported the compiled rest
coordinates, donor ledger and anatomy labels from that same build. Using an
older ledger with current coordinates had made the density comparison
meaningless. All geometry and transport candidates below are offline.

## Mass accounting

The 7,233-node, 28,632-cell original is the coarse mechanical shell.
Its compiled reference cell masses sum to 13.939679838734493 kg.
The compiled nodal ledger contains 9,644 typed rows. The geometry labels
separately identify the material, geometric donor body, anatomical region
and stable source face of each parent cell. The source materials are
verified against the compiler command's hash-pinned definitions.

The candidate route uses existing density-reserve allocation with a
sparse forward and reverse nearest-neighbor graph. It conserves every
source tet's mass and keeps routes within a single material and geometric
donor body. Sources that require no changes stay on identity routes.
Distances are evaluated in the compiled rest frame. The allocator proves
a conservative 75 mm route bound to every receiving child, then the
independent verifier checks the actual child-centroid distances, density,
per-source mass, received cell mass, bed coordinates and positivity in both
frames. Regenerated nodal dynamics and motion fidelity still require a native
build and accepted-state comparison. These cell budgets are inferred model
parameters.

The outer-only geometry is byte-for-byte equal between the dense and sparse
experiments. The 3,595,062-link dense allocation exceeded its 25 s limit.
The sparse graph allocated the same proposal in 10.746 s. All three
distinct fitted geometries conserve 13.939679838734492 kg within Float64
roundoff. Their maximum per-source residual is 1.735e−18 kg or less, the
longest independently checked child route is under 66 mm, and every cell is
positive in the compiled rest frame and the emitted Float32 pose. Existing
skin and fat density limits are retained. Some repaired columns exceed
their original thickness limits. The candidates are rejected geometry
experiments with verified cell mass.

## Exact crossings and limiting cells

| Mechanical shell | Outer skull | Skull interface | Deep skull | Whole shell against bones |
|---|---:|---:|---:|---:|
| Native initial | 702 | 2,311 | 2,685 | 17,878 |
| Outer footprint | 184 | 1,378 | 3,933 | 17,675 |
| All-layer footprint | 661 | 2,177 | 2,827 | 17,845 |
| All layers with cell 26915 held for diagnosis | 454 | 1,388 | 3,333 | 17,355 |

Counts use original shell-face and exact bone-triangle identities after
Float32 emission, with a geometry-identical split as control. The control
reproduces all 17,878 collision identities. The repairs did not add reported
self-intersections or crossings in other banks. All deep results are worse.
No geometry in the table passes the zero-crossing screen.

Enforcing inequalities for all six edges removes an approximation in the
linearized fit. In this scene its eight line searches yield identical
coordinates, topology and mass to the longest-edge comparison. Exact
trial checks reject the large moves on determinant and shape. They record
27 limiting child cells with typed original ancestry. Cell 26915 appears
59 times in the bounded witness list. Fixing its columns in the fit lets
more moves through and reproduces the remaining deep-layer tradeoff.
This was a restriction on the offline authoring search. The joints and
physical timestep have no new locks or forces.

The existing local topology screen retains four boundary-preserving flips
with improved minimum shape in both the original and fitted coordinates.
None includes cell 26915 while retaining material, donor, anatomical
region and stable source-face labels. The next reconstruction should
target the deep shell with a larger ancestry-aware cavity. The stored
witnesses distinguish the specific cells from unrelated skin geometry.

## Checks still open

The separate full-resolution source passed its volume gates after repairs
to 125 inverted and 154 low-quality cells. Its direct runtime transfer from
this coarse shell failed the prior motion checks. The latest retained
native prefix admits all 34 ligament rows, including both shoulders.
The real bed carried 0.400701 N at the endpoint against 952.864 N of body
weight, so the shoulder and tissue loads at supported equilibrium remain
unqualified. The same replay advanced 2 ms at 0.00002028× body-only.
The candidate geometries in this increment were kept out of that runtime.
Zero crossings in motion, the 16-second variable-pressure test and
end-to-end 0.1× remain open.

## Reproduction

Use the owning SSH Numi workspace and the pinned native executable in
the archived launch. Restore the exact registered skin, initial shell
and physical bone buffers from the preceding audit. Restore the rest and
mass ledgers from the private release or capture them with
`capture_reference.py`. The neighboring audit directories provide the
exact-predicate and subdivision code plus the earlier density-reserve
allocator. The JSON and SHA receipts in this directory retain the inputs,
failed trials and accepted one-root trace.

```sh
python3 rebuild.py outer
python3 rebuild.py complete-footprint --whole-skull
python3 rebuild.py whole-shape --whole-skull --all-edges
python3 screen_limiting_cavities.py whole-shape
python3 rebuild.py held-witness --whole-skull --all-edges --hold-parent 26915
python3 verify_rebuild.py outer complete-footprint whole-shape held-witness
```
