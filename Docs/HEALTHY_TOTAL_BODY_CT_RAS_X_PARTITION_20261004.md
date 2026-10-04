# Scan-specific adrenal and lung side partition — 2026-10-04

The scan-002 automatic `Adrenal-glands` and `Lung` labels previously crossed the
RAS-X midline and had three and six connected components respectively. Counting
components could not distinguish left from right, so this follow-up assigns
each occupied source voxel by the sign of its RAS-X center. RAS-X zero falls on
the voxel face at IJK x `255.5`; no occupied voxel center lies on the plane.

The preregistered plan predicted that both labels occupy each side and that the
partition conserves every source voxel. The independent source-stream audit
confirmed the prediction:

| Label | Source voxels | Left (positive RAS-X) | Right (negative RAS-X) | On plane |
| --- | ---: | ---: | ---: | ---: |
| Adrenal-glands | 2,490 | 1,119 | 1,371 | 0 |
| Lung | 1,716,761 | 964,043 | 752,718 | 0 |

The compiler emitted four exact voxel-boundary surfaces. The independent audit
re-read the compressed source NIfTI, reproduced the source hash and side counts,
reparsed each PLY, and recomputed edge topology, signed volume, RAS-X
half-space, source-label voxel envelope, and voxel-grid alignment. All four are
closed two-manifolds with zero nonmanifold vertices. Relative signed-volume
error versus side-specific voxel occupancy ranges from `0` to `5.54e-16`.

The retained plan, source-bound compiler receipt and independent audit are
[`plan-v2.json`](media/healthy-total-body-ct-ras-x-partition-20261004/plan-v2.json),
[`receipt.json`](media/healthy-total-body-ct-ras-x-partition-20261004/scan-002-run-v2/receipt.json),
and
[`independent-audit-v2.json`](media/healthy-total-body-ct-ras-x-partition-20261004/independent-audit-v2.json).
The four PLY candidates are stored beside the compiler receipt and bound by
its [`SHA256SUMS`](media/healthy-total-body-ct-ras-x-partition-20261004/scan-002-run-v2/SHA256SUMS)
manifest.

The inputs are the Healthy Total Body CTs v3 automatic segmentation release
(DOI `10.7937/NC7Z-4F76`, CC BY 4.0). The generated meshes remain attributed to
source labels 1 and 12 and are not presented as expert-corrected anatomy.

| Artifact | SHA-256 |
| --- | --- |
| Plan v2 | `2bcc8347c1ff38a9da0c79a11f0347816764b3eead1623a31c9eb38c846b56e9` |
| Compiler receipt v2 | `87cb2c4b1dd8697d4ec590728166625ec6ad65f9248f2f7d3d8a496e8dfc6bd6` |
| Independent audit v2 | `f8c57c3d4e88aa860bacf4c3dc41fb9b3d5d9c8b0676645deb532944493b0ecf` |

This resolves the scan-local geometric laterality gap for these two automatic
labels. It does not establish that the automatic masks are anatomically
accurate, that their boundaries are clinically correct, that scans are
registered to one another or to a Numi Human subject, or that organs have
mechanical ownership, contact, perfusion, or physiological behavior. No Numi
Human runtime anatomy or clinical-anatomy qualification is claimed.
