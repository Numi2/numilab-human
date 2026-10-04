# Scan-specific adrenal and lung laterality — 2026-10-04

In the Healthy Total Body CT v3 automatic labels, scan 002's `Adrenal-glands`
and `Lung` masks crossed the RAS-X midline and had three and six connected
components. Connected-component count alone could not identify left and right.
The follow-up partitions occupied source voxels by the sign of their RAS-X
center. For both scans, RAS-X zero lies on the voxel face at IJK x `255.5`.

The preregistered plans predicted that labels 1 (`Adrenal-glands`) and 12
(`Lung`) occupy both sides and that the partition conserves all source voxels.
An independent source-NIfTI stream confirmed both predictions:

| Scan | Label | Source voxels | Left (positive RAS-X) | Right (negative RAS-X) | On plane |
| --- | --- | ---: | ---: | ---: | ---: |
| 001 | Adrenal-glands | 1,786 | 1,150 | 636 | 0 |
| 001 | Lung | 1,476,822 | 826,286 | 650,536 | 0 |
| 002 | Adrenal-glands | 2,490 | 1,119 | 1,371 | 0 |
| 002 | Lung | 1,716,761 | 964,043 | 752,718 | 0 |

The two runs emitted eight exact voxel-boundary surfaces. Their independent
audits re-read the compressed NIfTI, reproduced source identity and side
counts, reparsed each PLY, and recomputed edge topology, signed volume, RAS-X
half-space, source-label voxel envelope, and voxel-grid alignment. All eight
surfaces are closed two-manifolds with zero nonmanifold vertices. Relative
signed-volume error versus side-specific voxel occupancy ranges from `0` to
`5.54e-16`.

The source-bound scan-001 plan, compiler receipt and independent audit are
[`plan-v3-scan-001.json`](media/healthy-total-body-ct-ras-x-partition-20261004/plan-v3-scan-001.json),
[`receipt.json`](media/healthy-total-body-ct-ras-x-partition-20261004/scan-001-run-v1/receipt.json),
and
[`independent-audit-scan-001-v1.json`](media/healthy-total-body-ct-ras-x-partition-20261004/independent-audit-scan-001-v1.json).
The corresponding scan-002 evidence is
[`plan-v2.json`](media/healthy-total-body-ct-ras-x-partition-20261004/plan-v2.json),
[`receipt.json`](media/healthy-total-body-ct-ras-x-partition-20261004/scan-002-run-v2/receipt.json),
and
[`independent-audit-v2.json`](media/healthy-total-body-ct-ras-x-partition-20261004/independent-audit-v2.json).
Each compiler receipt's `SHA256SUMS` binds its four PLY candidates.

Inputs come from Healthy Total Body CTs v3 (DOI `10.7937/NC7Z-4F76`, CC BY
4.0). The output keeps the source label identities and does not present the
automatic masks as expert-corrected anatomy. These are two scan identifiers;
this result does not establish separate participant identities or biological
replication.

This closes the scan-local geometric laterality gap for these labels. It does
not establish automatic-segmentation accuracy, clinically correct organ
boundaries, cross-scan registration, Numi Human subject binding, physical
tissue ownership, mechanics, perfusion, or physiology. No Numi Human runtime
anatomy or clinical-anatomy qualification is claimed.
