# Healthy Total Body CT segmentation source intake — 3 October 2026

Numi Human now has a reproducible, source-bound intake for the [TCIA Healthy
Total Body CT segmentation release](https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/).
The owner CLI verified all 30 NIfTI masks,
their ZIP CRCs and source archive identity, decoded every float32 voxel label,
joined the labels to the included workbook, and retained each scan's own
NIfTI RAS+ affine in the [intake receipt](media/healthy-total-body-ct-source-20261003/intake-v3.json).
The final v3 receipt SHA-256 is
`c0ce0ea7b3a068ddfbc90f56e079a77444571edeadd759ce72b35090342295e7`;
the immutable first-pass receipts remain beside it for audit history.

The registered Version 3 archive is 85,445,051 bytes with SHA-256
`f3daa9c5e9740c8d68688bd800f5d3107236099d3e319bbef04076aa4719b13f`. Its
30 masks are each `512 × 512 × 828`. The documented in-plane spacing is
`0.9765625 mm`, but scan `005` records `0.96484375 mm`; the importer preserves
that scan-specific value and affine rather than rescaling it. Five peripheral
bone labels are absent in some masks: carpal (26/30), metacarpal (24/30),
fingers (20/30), toes (26/30), and ulna (29/30).

The workbook contains 119 label identities. The masks use 36 nonbackground
label values; 83 workbook entries are absent from every mask. TCIA's dataset
description says “37 tissues,” so that count and the actual mask values remain
an explicit source discrepancy. The 36 observed labels include whole-organ
masks (including heart, brain, lung, kidney, liver, spleen and pancreas),
whole aorta and vena-cava masks, 20 bone labels, one combined skeletal-muscle
label, psoas, subcutaneous fat, and torso fat.

The source registry is
[`config/healthy-total-body-cts-source.v1.json`](../config/healthy-total-body-cts-source.v1.json),
and the importer is available as:

```sh
numi human healthy-total-body-ct-source \
  --archive Build/healthy-total-body-ct-20260923/healthy-total-body-ct-segmentations-v3.zip \
  --output Docs/media/healthy-total-body-ct-source-20261003/intake-v3.json
```

The receipt also records per-label raster-occupancy volume candidates as
`voxel_count × abs(det(affine))`; these are source segmentation geometry
measures and do not create physical tissue-volume or mass owners.

The full voxel audit requires the optional NumPy dependency; install the
`volume-ingest` extra in the Python 3.11+ environment used by the Human owner
CLI (`pip install -e '.[volume-ingest]'`).

TCIA lists the segmentation ZIP under CC BY 4.0 and requires the dataset
citation recorded in the receipt. This work downloaded only that segmentation
package; the separate DICOM CT collection is listed under the NIH Controlled
Data Access Policy and was not fetched. TCIA describes the masks as automatic
MOOSE segmentations at the 90-minute timepoint, and notes that the NIfTI masks
may need reorientation to match the CT images. This intake preserves each
published qform but has not checked voxel-to-CT overlay alignment.

This provides a second-cohort whole-body organ, bone, muscle and fat geometry
source for comparative registration and morphology checks. It remains a
separate cohort from Numi's current mechanical subject. The masks do not
establish expert segmentation accuracy or physical tissue volume ownership.
Subcutaneous fat does not supply a distinct skin layer; the combined muscle
label does not provide individual muscles, tendon paths or fascia. The aorta
and vena-cava masks do not distinguish lumen from wall or supply branch
connectivity, blood density, pressure transfer or perfusion. A heart label
does not provide myocardial layers, activation, conduction or beat mechanics.
No anatomy, material, physiology or clinical qualification is claimed.
