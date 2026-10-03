# Healthy Total Body CT Psoas source-mask components — 3 October 2026

The Psoas label was re-decoded from all 30 hash-pinned TCIA Healthy Total Body
CT Version 3 scans. Under 26-connectivity, 29 masks contain two components and
scan 024 contains three. The extra scan-024 component contains 37 voxels, or
0.0201% of its labelled mask; its centroid is retained in that scan's own
RAS+ frame. The two largest Psoas components have a median source-mask
occupancy balance ratio of 0.962 across the cohort.

The [corrected receipt](media/healthy-total-body-ct-psoas-components-20261003/corrected-receipt-v2.json)
records every component's voxel count, occupancy, voxel centroid and RAS+
centroid. Its [corrective plan](media/healthy-total-body-ct-psoas-components-20261003/corrective-plan-v2.json)
binds the exact intake, archive and predicate code, and identifies the first
receipt as superseded after a cohort-maximum calculation bug was caught. The
superseded receipt remains alongside the corrected version for audit history.

This is automatic source-mask topology and scan-local spatial evidence. It
does not determine whether the 37-voxel island is anatomy or a segmentation
defect, identify left and right muscles clinically, or establish individual
muscle surfaces, physical volume, Numi subject binding or mechanics.

Reproduce with the Human Python owner:

```sh
PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m numilab_human.healthy_total_body_ct_psoas_component_audit \
  --intake Docs/media/healthy-total-body-ct-source-20261003/intake-v4.json \
  --archive Build/healthy-total-body-ct-20260923/healthy-total-body-ct-segmentations-v3.zip \
  --plan Docs/media/healthy-total-body-ct-psoas-components-20261003/corrective-plan-v2.json \
  --output /private/tmp/healthy-total-body-ct-psoas-components.json
```
