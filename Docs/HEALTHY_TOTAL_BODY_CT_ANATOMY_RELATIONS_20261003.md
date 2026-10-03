# Healthy Total Body CT scan-local anatomy relations — 3 October 2026

The new Human owner audit checks four gross internal-organ centroid orderings
independently within each of the 30 TCIA Healthy Total Body CT segmentation
scans. It recomputes every selected RAS+ centroid through that scan's own
NIfTI affine before evaluating that `Brain`, `Heart`, `Lung`, and `Kidneys`
centroids lie superior to `Heart`, `Liver`, `Liver`, and `Bladder`, respectively.

All 120 relation rows pass, with no missing labels or failed orderings:

| Relation | Scans passing | RAS+ superior offset range |
| --- | ---: | ---: |
| Brain above heart | 30/30 | 302.34–399.52 mm |
| Heart above liver | 30/30 | 57.12–85.60 mm |
| Lungs above liver | 30/30 | 91.04–125.68 mm |
| Kidneys above bladder | 30/30 | 211.00–314.06 mm |

The source-bound [receipt](media/healthy-total-body-ct-anatomy-relations-20261003/receipt.json)
records each scan ID, affine hash, label IDs and voxel counts, centroids, and
measured offset. The audit keeps participant frames separate and binds the
exact spatial intake receipt. The capability registry now exposes these
cross-scan facts under whole-body anatomy and internal organs.

This is a coarse external automatic-segmentation consistency check. It does
not verify voxel-to-DICOM alignment or segmentation accuracy, register the
cohort to Numi's mechanical subject, or establish organ surfaces, volume
ownership, mechanics, physiology, or clinical anatomy. The external-source
subject-binding and organ-mechanics gates remain open.

Reproduce the audit with the Human Python owner:

```sh
PYTHONPATH=src:Sources/myosim/checkout \
  .venv-mujoco312/bin/python -m numilab_human.healthy_total_body_ct_anatomy_relations \
  --intake Docs/media/healthy-total-body-ct-source-20261003/intake-v4.json \
  --output /private/tmp/healthy-total-body-ct-anatomy-relations.json
```
