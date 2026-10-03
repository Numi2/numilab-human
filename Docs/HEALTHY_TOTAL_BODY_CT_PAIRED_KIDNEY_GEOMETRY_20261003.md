# Healthy Total Body CT paired-kidney source-mask geometry — 3 October 2026

The Human owner re-decoded the aggregate Kidney label from all 30 scans in the
hash-pinned TCIA Healthy Total Body CT Version 3 archive. Every scan retained
exactly two components under 26-connectivity, and the component centroids were
laterally separated after transformation through that scan's own NIfTI RAS+
affine.

| Measurement | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| RAS+ X separation between ordered component centroids | 105.26 mm | 126.42 mm | 168.17 mm |
| Smaller/larger source-mask occupancy ratio | 0.800 | 0.947 | 0.997 |
| Combined kidney-label voxel occupancy | 180.35 mL | 306.55 mL | 486.67 mL |

The immutable [receipt](media/healthy-total-body-ct-paired-kidney-20261003/receipt.json)
contains every per-scan component count, voxel centroid, RAS+ centroid,
occupancy, affine, and member hash. The [registered plan](media/healthy-total-body-ct-paired-kidney-20261003/preregistered-plan-v1.json)
binds the exact source intake, archive, and predicate code. The measurement is
automatic segmentation-mask occupancy; it is not a clinical organ volume or
expert confirmation of segmentation accuracy. The lower/higher RAS+ X labels
describe coordinate ordering only.

This adds gross paired-organ source evidence. It does not register the cohort
to Numi's subject or establish kidney surfaces, physical tissue volume,
mechanics, or physiology.

Reproduce with the Human Python owner:

```sh
PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m numilab_human.healthy_total_body_ct_paired_kidney_audit \
  --intake Docs/media/healthy-total-body-ct-source-20261003/intake-v4.json \
  --archive Build/healthy-total-body-ct-20260923/healthy-total-body-ct-segmentations-v3.zip \
  --plan Docs/media/healthy-total-body-ct-paired-kidney-20261003/preregistered-plan-v1.json \
  --output /private/tmp/healthy-total-body-ct-paired-kidney.json
```
