# Whole-body source-frame overlap census — 2 October 2026

The [final census receipt](media/whole-body-source-overlap-census-20261002/receipt-v4.json)
compares all **198 of 198** compiled organ crossing pairs whose raw BodyParts3D
members have a pinned source-body and compiled rigid-owner frame. It reads 104
hash-verified OBJ meshes from both locked BodyParts3D archives. The compiled
triangle indices are mapped through the exact source-face provenance in the
topology-repair manifest before comparing witnesses.

For 187 pairs, the raw and compiled meshes have matching crossing status,
intersecting triangle-pair count, and first-12 source-face witness sample
directly. Ten additional pairs match after applying the exact source-face maps
for topology-repaired surfaces. In the final pair, FJ2821/FJ2824, the raw
source has 185 intersecting triangle pairs and the compiled surfaces have 183.
The two missing witnesses, `[10347, 15941]` and `[10350, 15941]`, each use one
of two opposite duplicate FJ2821 faces removed by the recorded topology repair;
the mapped first-12 samples still agree. Thus all 198 crossing pairs have
source-supported witnesses, and no unexplained source/compiled difference
remains in this comparison.

Across those pairs the raw-source count is 34,806 triangle-pair intersections
and the compiled count is 34,804. The largest raw-versus-compiled difference
in reported intersection-segment length is 0.101312 mm, for FJ2577/FJ2605
(13.299690 mm raw and 13.401002 mm compiled). Triangle intersections and
segment lengths do not measure penetration depth or overlap volume.

This census localizes the measured crossings to pinned source faces, including
the documented duplicate-face cleanup. It does not decide whether a crossing
is an intended regional seam, establish clinical anatomy, tissue ownership,
contact mechanics, or qualification. No source geometry or registration was
changed; the eight individually unqualified organ surfaces remain excluded
from the compiled overlap diagnostic.

The receipt binds both BodyParts3D archive hashes, the source-family manifests,
baseline owner map, source semantics, topology-repair manifest, corrected
compiled-overlap receipt, every compared raw-member hash, and the census
implementation. Reproduce from the repository root:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv-mujoco312/bin/python \
  -m numilab_human.whole_body_source_overlap_census \
  --output /private/tmp/whole-body-source-overlap-census.json
```

The output is deterministic and can be compared byte-for-byte with the retained
`receipt-v4.json`.
