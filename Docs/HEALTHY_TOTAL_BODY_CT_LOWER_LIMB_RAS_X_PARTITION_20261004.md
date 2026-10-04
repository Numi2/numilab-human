# Scan-specific lower-limb left/right partition — 2026-10-04

The Healthy Total Body CT v3 scan-001 automatic masks for the femur, fibula,
patella, and tibia were split at the exact RAS-X midplane. Positive NIfTI RAS-X
was assigned to the subject's right and negative RAS-X to the subject's left.
The preregistered plan predicted bilateral occupancy for all four labels and
exact source-voxel conservation.

| Label | Source voxels | Left (negative RAS-X) | Right (positive RAS-X) | On plane |
| --- | ---: | ---: | ---: | ---: |
| Femur | 596,925 | 297,851 | 299,074 | 0 |
| Fibula | 70,890 | 35,282 | 35,608 | 0 |
| Patella | 25,791 | 13,245 | 12,546 | 0 |
| Tibia | 342,569 | 171,230 | 171,339 | 0 |

All eight side-specific voxel-boundary meshes independently pass source-NIfTI
identity and voxel recount, exact partition, closed two-manifold topology,
source-grid and envelope checks, and signed-volume comparison to their own
side's occupied voxels. The maximum relative volume error is `1.48e-15`.
See the [preregistered plan](media/healthy-total-body-ct-lower-limb-ras-x-partition-20261004/plan-scan-001-v1.json),
[compiler receipt](media/healthy-total-body-ct-lower-limb-ras-x-partition-20261004/scan-001-run-v1/receipt.json),
[independent audit](media/healthy-total-body-ct-lower-limb-ras-x-partition-20261004/independent-audit-scan-001-v1.json),
and [file checksums](media/healthy-total-body-ct-lower-limb-ras-x-partition-20261004/SHA256SUMS).

The scan's separate patella anteriority audit finds the automatic patella label
in front of the femur's anterior envelope at all `1,730` shared projections;
the minimum offset is `9.766 mm` and the median is `23.438 mm` ([audit](media/healthy-total-body-ct-surface-20261003/patella-anteriority-audit-v1.json)).
This source-scan result supports left/right assignment and gross anterior
placement in that scan. It does not verify automatic segmentation accuracy,
participant identity, registration to Numi Human, patellar tracking under
flexion or load, cartilage/contact, clinical anatomy, or physiology.

The partition code uses the NIfTI RAS+ convention documented in the
[NIfTI coordinate specification](https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform.html/document_view.html).
