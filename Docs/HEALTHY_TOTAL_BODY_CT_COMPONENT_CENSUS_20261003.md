# Healthy Total Body CT source-mask component census — 3 October 2026

The Human owner audit re-decoded 15 organ, vessel, and soft-tissue label masks
from all 30 scans in the hash-pinned Healthy Total Body CT Version 3
segmentation archive. It measured connected components twice for every
scan-label unit: face-sharing 6-connectivity and face/edge/corner 26-connectivity.
All 450 selected voxel counts matched the source intake and all 30 uncompressed
NIfTI member hashes matched. The [preregistered plan](media/healthy-total-body-ct-component-census-20261003/preregistered-plan-v1.json)
binds the source archive, intake, label IDs, and predicate code.

| Source label | 6-connected components, min/median/max | 26-connected components, min/median/max | 26-connected single-component scans |
| --- | ---: | ---: | ---: |
| Aorta | 1 / 1 / 1 | 1 / 1 / 1 | 30/30 |
| VCI | 1 / 1 / 2 | 1 / 1 / 2 | 29/30 |
| Bladder | 1 / 1 / 1 | 1 / 1 / 1 | 30/30 |
| Brain | 1 / 1 / 2 | 1 / 1 / 2 | 28/30 |
| Heart | 1 / 1 / 3 | 1 / 1 / 2 | 29/30 |
| Kidneys (aggregate label) | 2 / 2 / 2 | 2 / 2 / 2 | 0/30 |
| Liver | 1 / 1 / 2 | 1 / 1 / 2 | 29/30 |
| Lung (aggregate label) | 1 / 1 / 6 | 1 / 1 / 6 | 25/30 |
| Pancreas | 1 / 1 / 1 | 1 / 1 / 1 | 30/30 |
| Spleen | 1 / 1 / 1 | 1 / 1 / 1 | 30/30 |
| Thyroid | 1 / 1 / 2 | 1 / 1 / 2 | 28/30 |
| Psoas | 2 / 2 / 6 | 2 / 2 / 3 | 0/30 |
| Skeletal-muscle (combined label) | 215 / 469 / 1,328 | 65 / 109 / 306 | 0/30 |
| Subcutaneous-fat | 682 / 1,188 / 2,143 | 277 / 467 / 859 | 0/30 |
| Torso-fat | 519 / 1,023 / 1,671 | 242 / 397.5 / 748 | 0/30 |

The registered prediction for Aorta and VCI was met: Aorta is one 26-connected
mask in all scans and VCI in 29/30. The prediction for the aggregate kidneys and
lungs to remain at one or two components in at least 27 scans was only partly
met: kidneys are two components in all 30 scans, while lungs are one or two in
26/30. The remaining lung masks have 3, 4, 6, and 3 components in scans 004,
002, 026, and 031 respectively. Their largest-component fractions are still
0.999 or greater, so these extra components occupy only a small fraction of
their label voxels.

The combined skeletal-muscle mask is highly fragmented by this adjacency
measure, with a median of 109 26-connected components, while its largest
component contains at least 99.4% of labeled voxels in every scan. The separate
Psoas label is two components in 29 scans and three in scan 024 under
26-connectivity. Subcutaneous and torso fat are not skin labels; this dataset
still supplies no distinct skin layer.

The raw rows and summaries are in the source-bound [receipt](media/healthy-total-body-ct-component-census-20261003/receipt.json).
The 30-participant cohort and segmentation release are described by
[TCIA's Healthy Total Body CT collection](https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/).

This is automatic source-mask topology evidence. Voxel connectivity does not
establish segmentation accuracy, a vascular lumen or branch graph, a physical
tissue volume, Numi subject binding, mechanics, physiology, or clinical
anatomy. No whole-body capability gate is closed by this census.

Reproduce with the Human Python owner:

    PYTHONPATH=src:Sources/myosim/checkout \
      .venv-mujoco312/bin/python -m numilab_human.healthy_total_body_ct_component_census \
      --intake Docs/media/healthy-total-body-ct-source-20261003/intake-v4.json \
      --archive Build/healthy-total-body-ct-20260923/healthy-total-body-ct-segmentations-v3.zip \
      --plan Docs/media/healthy-total-body-ct-component-census-20261003/preregistered-plan-v1.json \
      --output /private/tmp/healthy-total-body-ct-component-census.json
