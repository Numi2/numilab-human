# Corrected scan-specific adrenal and lung laterality — 2026-10-04

The first published CT RAS-X partition artifacts assigned the left and right
names backwards. The implementation had treated positive NIfTI RAS-X as left.
NIfTI RAS+ uses positive X for the subject's right and negative X for the
subject's left ([NIfTI coordinate documentation](https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform.html/document_view.html)).
The v1 plans, receipts, meshes, and audits remain preserved in the original
artifact directory and are superseded for laterality claims by the corrected
schema-v2 evidence linked below. The [correction record](media/healthy-total-body-ct-ras-x-partition-side-correction-20261004/method-correction-v1.json)
binds the superseded and corrected receipts and audits.

In the Healthy Total Body CT v3 automatic labels, both scans' `Adrenal-glands`
and `Lung` masks cross the RAS-X midline. Connected-component counts do not
identify left and right. These corrected partitions assign every occupied
source voxel by the sign of its RAS-X center. For both scans, RAS-X zero lies
on the voxel face at IJK x `255.5`.

The preregistered plans predicted that labels 1 (`Adrenal-glands`) and 12
(`Lung`) occupy both sides and that the partition conserves all source voxels.
Independent source-NIfTI streams confirmed both predictions:

| Scan | Label | Source voxels | Left (negative RAS-X) | Right (positive RAS-X) | On plane |
| --- | --- | ---: | ---: | ---: | ---: |
| 001 | Adrenal-glands | 1,786 | 636 | 1,150 | 0 |
| 001 | Lung | 1,476,822 | 650,536 | 826,286 | 0 |
| 002 | Adrenal-glands | 2,490 | 1,371 | 1,119 | 0 |
| 002 | Lung | 1,716,761 | 752,718 | 964,043 | 0 |

The final-source rebuilds emitted eight exact voxel-boundary surfaces. The
independent audits re-read the compressed NIfTI, reproduced source identity
and side counts, reparsed each PLY, and recomputed edge topology, signed
volume, RAS-X half-space, source-label voxel envelope, and voxel-grid
alignment. All eight surfaces are closed two-manifolds with zero nonmanifold
vertices. Relative signed-volume error versus side-specific voxel occupancy
ranges from `0` to `5.54e-16`.

For scan 001, see the corrected [v2 plan](media/healthy-total-body-ct-ras-x-partition-side-correction-20261004/plan-scan-001-v2.json), [compiler receipt](media/healthy-total-body-ct-ras-x-partition-side-correction-20261004/scan-001-run-v2/receipt.json), and [independent audit](media/healthy-total-body-ct-ras-x-partition-side-correction-20261004/independent-audit-scan-001-v2.json). For scan 002, see its [v2 plan](media/healthy-total-body-ct-ras-x-partition-side-correction-20261004/plan-scan-002-v2.json), [compiler receipt](media/healthy-total-body-ct-ras-x-partition-side-correction-20261004/scan-002-run-v2/receipt.json), and [independent audit](media/healthy-total-body-ct-ras-x-partition-side-correction-20261004/independent-audit-scan-002-v2.json). Per-run `SHA256SUMS` bind each receipt to its four PLY candidates; the correction directory's `SHA256SUMS` binds all its evidence files except the manifest itself.

Inputs come from Healthy Total Body CTs v3 (DOI `10.7937/NC7Z-4F76`, CC BY
4.0). The output retains the source label identities and does not present the
automatic masks as expert-corrected anatomy. These are two scan identifiers;
this result does not establish separate participant identities or biological
replication.

This closes the scan-local geometric laterality gap for these labels. It does
not establish automatic-segmentation accuracy, clinically correct organ
boundaries, cross-scan registration, Numi Human subject binding, physical
tissue ownership, mechanics, perfusion, or physiology. No Numi Human runtime
anatomy or clinical-anatomy qualification is claimed.
