# Source-bound patellofemoral intersection loop - 30 September 2026

The earlier [exact audit](PATELLOFEMORAL_CONTACT_INTERSECTIONS_20260930.md)
found 18 crossing triangle pairs in the pinned Open Knee(s) source and both
compiled knees. The [intersection-loop artifact](media/patellofemoral-loop-20260930/receipt.json)
resolves their topology and records an exact geometric witness for each pair.
The 18 segments meet at 18 unique points in **one closed, degree-two loop**.
Its length is **2.240 mm** in the source and **2.205 mm** in either compiled
placement. Every segment endpoint has barycentric coordinates on the authored
patellar and femoral triangles; rebuilding either compiled point from those
weights differs by at most 1 femtometre in Float64. The intersecting face
normals are nearly opposed on both sides (compiled dot products from
**-0.99983 to -0.99149**).

This is a concrete initial-contact geometry input. The artifact names the
exact face pair, global compiled node indices, ordered loop connectivity,
point positions and both sets of barycentric weights. A native solver can use
that information to seed triangle-pair constraints without treating the entire
22,478-face femoral source surface as one simultaneously active patch. The
source and compiled mesh coordinates, faces and payload hashes remain
unchanged.

**No loaded contact is qualified by this artifact.** A closed intersection
curve does not determine a unique separating displacement, penetration depth,
pressure, material response or accepted-state energy. The next native step
must define and test those quantities under a source-bound initial-contact
policy, then integrate the resulting constraints into the owning contact/FEM
transaction with rollback and sustained load evidence. The current
[CPU preflight](PATELLOFEMORAL_NATIVE_PREFLIGHT_SCOPE_20260930.md) still fails
localized initial-contact admission.

Reproduce the geometry artifact from the retained source and bilateral
payloads:

```sh
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python -m tools.build_patellofemoral_intersection_loop
```
