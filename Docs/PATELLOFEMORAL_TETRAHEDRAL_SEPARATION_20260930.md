# Exact solid separation in the retained patellofemoral candidate

The earlier [full native cartilage preflight](media/patellofemoral-full-native-20260930/README.md)
showed zero surface crossings after an **unadopted 20 µm patellar-cartilage
pose change**, but its femoral cartilage boundary has a two-fan vertex. A
surface-parity sample therefore could not establish that the two cartilage
solids were disjoint.

The [tetrahedral separation receipt](media/patellofemoral-tetrahedral-separation-20260930/receipt.json)
tests the retained **left/right × contact-on/off** native accepted positions
directly. It maps all **121,105 patellar** and **87,072 femoral** source
tetrahedra to their exact binary32 native positions. An integer-grid search
finds all **5,512** overlapping tetrahedron AABBs in each state. For each of
those pairs, exact integer face-normal and edge-cross-edge separating-axis
predicates find **zero boundary contacts and zero interior overlaps**. Every
tetrahedron has positive exact oriented volume after the compiled right-mirror
corner swap. Six synthetic
predicate controls include a skew edge-edge separation that face normals
alone miss.

This closes **PTC-versus-FMC solid separation for those four retained native
states** without assuming that the FMC boundary is a manifold. It does not
repair the source's original 18 surface crossings or the femoral two-fan
vertex, and the 20 µm pose remains unadopted. The native step uses synthetic
cartilage material and density, excludes the patellar bone and QAT/PTL
attachments, and supplies a prescribed approach rather than physiological
load. The contact-on run has 11 active histories but zero recorded barrier
impulse. Cartilage pressure, loaded force/energy transfer, sustained flexion,
clinical anatomy and whole-body standing remain unqualified.

Reproduce without rerunning Metal, using the four hashed accepted-position
buffers, the two hash-pinned `NHCAR1` inputs in
`Build/full-patellofemoral-matter-20260930`, and the pinned Open Knee(s)
source. The verifier compares every native input tetrahedron to source order
and the compiled right-mirror parity before testing the accepted states:

```sh
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/audit_patellofemoral_tetrahedral_separation.py
```
