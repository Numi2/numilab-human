# Healthy Total Body CT source spatial candidates — 3 October 2026

The source intake now records per-label spatial location and extent for the 30
TCIA masks in addition to their exact source hashes, voxel counts, occupancy
volumes and independent NIfTI affines. The immutable
[v2-schema receipt](media/healthy-total-body-ct-source-20261003/intake-v4.json)
contains 1,055 observed label/scan geometry candidates across 36 labels. Its
SHA-256 is `b23e5d88b9bbd0f1e017045ddae0f9f2d69a3c302b3435a24c66ee0f6ed1ea21`;
the pinned archive remains SHA-256
`f3daa9c5e9740c8d68688bd800f5d3107236099d3e319bbef04076aa4719b13f`.

Each observed label records its voxel count, arithmetic centroid of occupied
voxel centres in IJK, inclusive minimum/maximum voxel-centre indices, the
centroid transformed through that scan's NIfTI affine to RAS millimetres, and
the RAS axis-aligned envelope of the rectangular voxel-bound prism. The
envelope is computed from the eight half-voxel-expanded corners, so it bounds
the labelled voxels but is not a surface reconstruction or a tight mask-shape
description. Every scan keeps its own affine and source frame.

The full pass revalidated all 30 registered NIfTI members. All pre-existing
v3 scan hashes, voxel counts, voxel-volume candidates, affines, label coverage
and source metadata reproduce exactly. Across the new spatial records,
voxel counts match the integer-label audit; all indices lie within the source
array; centroids reproduce through the affine, remain within their recorded
envelopes, and are finite.

These records extend source anatomy toward spatial registration of whole-body
organs, major vessels, bones, muscle and fat. RAS coordinates from different
participants are not treated as a shared anatomical frame. The intake still
has no verified NIfTI-to-DICOM overlay or expert segmentation review, and the
masks remain a separate cohort from Numi's mechanical subject. No physical
tissue volume, mass, skin layer, vascular lumen, tissue mechanics, perfusion,
electrical activity or clinical qualification is inferred from the spatial
candidates.

Reproduce the receipt with the Human owner CLI:

```sh
numi human healthy-total-body-ct-source \
  --archive Build/healthy-total-body-ct-20260923/healthy-total-body-ct-segmentations-v3.zip \
  --output /private/tmp/healthy-total-body-ct-source-spatial-v4.json
```

The v3 intake remains unchanged as the initial inventory snapshot.

The follow-up [source-named cohort summary](HEALTHY_TOTAL_BODY_CT_COHORT_SUMMARY_20261003.md)
joins observed mask values to the exact workbook label names and reports
per-label coverage and affine-scaled voxel-occupancy distributions without
pooling participant coordinate frames.

The follow-up [voxel-boundary surface candidates](HEALTHY_TOTAL_BODY_CT_SURFACE_CANDIDATES_20261003.md)
cover scan-001 whole-organ, major-vessel and skeletal masks. The Patella
source label is anterior to the distal Femur in 1,730 shared source-coordinate
projections, but these data do not register to Numi Human. Many automatic bone
and organ labels contain nonmanifold contacts in their exact voxel unions.
