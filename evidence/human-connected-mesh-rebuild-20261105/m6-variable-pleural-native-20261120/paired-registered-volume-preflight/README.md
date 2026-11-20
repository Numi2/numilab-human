# Repaired skin and fat volume admitted by the production registration reader

The Apple M6 host `numi` imported the 311,020 nodes, 1,448,804 tetrahedra, and 14 stable deep attachments through the native registered-volume importer. It used the retained 157-body pose and existing 86 donor frames. The repaired cells pass the unchanged normalized volume and absolute volume policies. The locally changed material faces have zero exact crossings against the full 833,108-face boundary bank at the retained reference pose.

One inner-surface node moved 7.45 nm and two interior skin nodes moved at most 4.271 µm. The element table and its ordering were preserved. The 180 authored head wall faces required exact source ancestry to be propagated through the same winning P1 tuples and recorded on the current cell rows. This preserves the full P1 ancestry and the skin and fat region and donor assignments.

The volume and registration controls record two rejected precursors. The volume repairs use the existing physical thresholds. This evidence covers static registration and local reference-pose crossing checks. Complete native motion, whole-body crossings, equilibrium and performance require separate runs.

Use the retained checkout and asset paths in the receipts. The offline harness calls production readers. Build and run on SSH `numi`.

```bash
NUMI_HUMAN_REGISTERED_MESH_OVERRIDE="$PWD/absolute-volume-repaired-tetgen" \
NUMI_HUMAN_REGISTERED_SIDECAR_OVERRIDE="$PWD/paired-repaired-registration-floor" \
NUMI_HUMAN_REGISTERED_FACETS_OVERRIDE="$PWD/paired-repaired-registration-floor/facet-ancestry-current.csv" \
NUMI_HUMAN_REGISTERED_LABELS_OVERRIDE="$PWD/paired-floor-child.labels.csv" \
  /bin/bash run_repaired_registered_volume_check.sh
```
