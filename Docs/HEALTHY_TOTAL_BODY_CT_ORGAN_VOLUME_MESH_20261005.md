# Healthy total-body CT organ volume mesh candidates — 2026-10-05

The automatic organ and vessel labels from scans 001 and 002 now have source-mask-bound tetrahedral volume candidates. Across the two scans, 7,350,007 voxels were decomposed into 44,100,042 positive tetrahedra. The source NIfTI hash and each label count matched the intake for both scans. Independent verifiers reread both scans and all serialized VTK meshes, checked tetrahedra and face incidences, matched each extracted boundary to its exact source mask, and measured relative per-label volume error from `1.13e-16` through `3.38e-15`.

| Scan-001 label | Source voxels | Tetrahedra | Voxel volume candidate (mL) |
|---|---:|---:|---:|
| Lung | 1,476,822 | 8,860,932 | 3,301.307 |
| Brain | 659,831 | 3,958,986 | 1,474.995 |
| Liver | 524,736 | 3,148,416 | 1,173.002 |
| Heart | 260,376 | 1,562,256 | 582.048 |
| Kidneys | 157,014 | 942,084 | 350.991 |
| Aorta | 124,336 | 746,016 | 277.942 |
| Bladder | 61,152 | 366,912 | 136.700 |
| Spleen | 41,346 | 248,076 | 92.425 |
| VCI | 27,457 | 164,742 | 61.378 |
| Pancreas | 24,818 | 148,908 | 55.478 |
| Thyroid | 5,904 | 35,424 | 13.198 |
| Adrenal glands | 1,786 | 10,716 | 3.992 |

Scan 002 independently passes the same direct-mask pipeline:

| Scan-002 label | Source voxels | Tetrahedra | Voxel volume candidate (mL) |
|---|---:|---:|---:|
| Lung | 1,716,761 | 10,300,566 | 3,837.669 |
| Brain | 681,388 | 4,088,328 | 1,523.183 |
| Liver | 778,940 | 4,673,640 | 1,741.252 |
| Heart | 322,608 | 1,935,648 | 721.162 |
| Kidneys | 177,391 | 1,064,346 | 396.542 |
| Aorta | 96,992 | 581,952 | 216.817 |
| Bladder | 23,843 | 143,058 | 53.299 |
| Spleen | 107,936 | 647,616 | 241.282 |
| VCI | 32,039 | 192,234 | 71.620 |
| Pancreas | 38,495 | 230,970 | 86.052 |
| Thyroid | 5,546 | 33,276 | 12.398 |
| Adrenal glands | 2,490 | 14,940 | 5.566 |

The initial surface flood-fill method was rejected because an enclosed four-voxel void was filled as solid. Direct mask ingestion reproduces the registered source surface boundary without filling that void. Both the failed fill and the first verifier's face-ordering error remain recorded with the [candidate artifacts](media/healthy-total-body-ct-organ-volume-tet-candidates-20261005/).

These are scan-specific geometry candidates from automatic segmentations, not clinical anatomy. Expert review, independent-participant identity and replication, Numi Human subject binding, vertex-link manifold qualification, physical organ ownership, calibrated tissue materials and mass, vessel lumen/perfusion, mechanics, and physiological validation remain open. The [scan-001 independent receipt](media/healthy-total-body-ct-organ-volume-tet-candidates-20261005/attempt-002/independent-verification.json) and [scan-002 independent receipt](media/healthy-total-body-ct-organ-volume-tet-candidates-20261005/scan-002/independent-verification.json) carry exact verifier measurements and provenance.
