# Kidney voxel-tetrahedron geometry candidate

This evidence package expands the four scan-specific left/right kidney
surfaces into source-voxel tetrahedral meshes. Attempt 002 is recorded by the
root `plan.json` and `meshes/receipt.json`. Attempt 001's rejected partial
mesh, exact plan, and failure record are retained under `attempt-001/`.

The compiler reads the pinned CT source intake, the bilateral surface export,
the source export result, and its separately checksummed current verification.
The old surface receipt's embedded verification digest does not match that
current verification file; the discrepancy is retained in the plan and
compiler receipt. The independent volume-mesh verifier reparses the final VTK
files and directly compares their boundary triangle multisets to the source
PLYs.

Run from the repository root:

```sh
numi human healthy-total-body-ct-voxel-tets \
  --plan Docs/media/bilateral-kidney-voxel-tet-candidate-20261004/plan.json \
  --output Docs/media/bilateral-kidney-voxel-tet-candidate-20261004/meshes
python3 Docs/media/bilateral-kidney-voxel-tet-candidate-20261004/verify_candidate.py
```

Attempt 002 emitted four compressed VTK unstructured grids with 2,006,430
positive tetrahedra. Independent serialized-mesh verification passed for
tetrahedron orientation, face incidence, volume closure, and exact source
surface boundary recovery. These are still scan-specific automatic-segmentation
geometry candidates; the receipt explicitly leaves subject binding, material,
mass ownership, mechanics, perfusion, and physiology unqualified.
