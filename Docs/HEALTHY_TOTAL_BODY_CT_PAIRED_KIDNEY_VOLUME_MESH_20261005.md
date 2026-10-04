# Paired-kidney voxel volume meshes — 2026-10-05

The four source-bound kidney surfaces from scans 001 and 002 now have matching
voxel-domain tetrahedral geometry candidates. The owner command reconstructs
the source voxel cells adjacent to each oriented boundary, fills the closed
interior by six-connected exterior flood fill, and partitions every occupied
voxel into six Freudenthal tetrahedra. This preserves the stair-stepped source
boundary and uses each scan's own NIfTI RAS+ affine; output VTK point units are
metres.

| Scan | Side | Source voxels | Tetrahedra | Occupancy volume | Relative volume error |
| --- | --- | ---: | ---: | ---: | ---: |
| 001 | Left | 81,746 | 490,476 | 182.736 ml | `1.48e-16` |
| 001 | Right | 75,268 | 451,608 | 168.255 ml | `1.61e-16` |
| 002 | Left | 88,565 | 531,390 | 197.979 ml | `9.58e-16` |
| 002 | Right | 88,826 | 532,956 | 198.563 ml | `5.46e-16` |

The separate standard-library verifier parses every serialized VTK point and
tetrahedron, confirms positive signed volume and valid one-or-two tetrahedral
face incidence, extracts the tetrahedral boundary, and compares its exact
voxel-grid triangle multiset with the source PLY. All four boundaries match;
all 2,006,430 tetrahedra are positive. Compiler and verifier records are under
[the candidate evidence directory](media/bilateral-kidney-voxel-tet-candidate-20261004/).
The per-mesh payloads are compressed VTK unstructured grids in
[the meshes directory](media/bilateral-kidney-voxel-tet-candidate-20261004/meshes/).

Attempt 001 and its partial mesh remain preserved: ordinary summation produced
a `1.20e-11` relative error and correctly failed the unchanged `1e-12` gate.
Attempt 002 uses compensated summation and passes. The earlier surface-export
receipt's embedded verification digest does not match the current separately
checksummed verification file; that inconsistency is recorded in the new plan
and receipt, and the current verification independently binds the export
result, source surfaces, and plan.

These are source-voxel geometry candidates from automatic MOOSE segmentations,
not expert-reviewed organs or a qualified human model. The two scan IDs are not
asserted to represent independent participants. The meshes have no Numi Human
subject binding, organ physical-volume authority, mass owner, tissue material,
mechanics admission, vascular lumen, perfusion, physiological qualification,
or clinical validation. VTK serialization also does not establish import into
the native Matter runtime.

Reproduce with:

```sh
numi human healthy-total-body-ct-voxel-tets \
  --plan Docs/media/bilateral-kidney-voxel-tet-candidate-20261004/plan.json \
  --output Docs/media/bilateral-kidney-voxel-tet-candidate-20261004/meshes
python3 Docs/media/bilateral-kidney-voxel-tet-candidate-20261004/verify_candidate.py
```
