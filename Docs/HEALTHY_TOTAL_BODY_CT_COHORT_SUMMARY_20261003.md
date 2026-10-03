# Source-named CT cohort geometry summary — 3 October 2026

The new Human owner command maps the validated mask-level spatial candidates
to their exact TCIA workbook names and reports label coverage and per-label
voxel-occupancy distributions over the 30 scans. Its immutable
[version 2 cohort summary receipt](media/healthy-total-body-ct-cohort-20261003/v2/cohort.json)
binds the input v4 intake by SHA-256
`b23e5d88b9bbd0f1e017045ddae0f9f2d69a3c302b3435a24c66ee0f6ed1ea21` and the
segmentation archive by SHA-256
`f3daa9c5e9740c8d68688bd800f5d3107236099d3e319bbef04076aa4719b13f`. The
summary receipt SHA-256 is
`d712dfb008fecfe6987a7d520baafdc7b6d9f86989762c7cd53cd5545cb9e25b`. Its
[run manifest](media/healthy-total-body-ct-cohort-20261003/v2/run-manifest.json)
and [checksums](media/healthy-total-body-ct-cohort-20261003/v2/SHA256SUMS)
bind the owner CLI invocation and the source/audit files.

It reconciles all 1,055 observed label/scan records against voxel counts,
affine-scaled raster occupancy, voxel-centre bounds, RAS+ centroids, and
world-space voxel-envelope bounds. It preserves the 119 source label names,
including the 83 not observed in any mask, and reports each scan's own affine.
Median and quartiles use linear interpolation at `(n - 1) * probability` over
the scans where that label is present. Participant coordinates are not pooled
because the source scans have not been registered to each other.

Examples from the exact source dictionary include heart-label occupancy with
a 508.35 mL median and 294.46–721.16 mL observed range; aggregate `Kidneys`
with a 306.55 mL median and 180.35–486.67 mL range; aggregate `Lung` with a
3,047.62 mL median and 1,690.81–4,760.26 mL range; `Aorta` with a 184.67 mL
median and 76.81–293.41 mL range; and `VCI` with a 61.68 mL median and
34.70–114.61 mL range. These are affine-scaled **voxel-occupancy candidates**
from automatically generated masks, not qualified anatomical or physical
tissue volumes. The kidney and lung labels remain the source's aggregates;
they are not split into laterality or lobes.

Coverage is 30/30 for the reported whole-organ and vessel labels. Peripheral
labels are missing in some scans: carpal 26/30, metacarpal 24/30, fingers
20/30, toes 26/30, and ulna 29/30. The receipt retains each missing scan ID.

Reproduce using the owning Human CLI:

```sh
numi human healthy-total-body-ct-cohort \
  --receipt Docs/media/healthy-total-body-ct-source-20261003/intake-v4.json \
  --output /private/tmp/healthy-total-body-ct-cohort-v1.json
```

This adds source-named cohort statistics for comparative anatomy and later
registration. It does not reread or independently review the segmentations,
cross-register participants, bind the cohort to Numi's current subject, map
labels to a HumanPack ontology, infer skin or vascular lumens, or assign
physical volume, mass, materials, mechanics, physiology, or clinical validity.
