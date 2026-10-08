# Selected lung composition 1113

This is a source-bound, offline composition candidate for the selected 1078 lung repairs. It is not a final anatomy or native-cycle qualification. The native preflight launcher consumes the immutable v8 output separately; exact full cross-surface scans and native accepted-cycle review remain independent gates.

The recipe applies the selected 306 collapse, the paired 307/309 inferred-reference diagonal flip, two sequential shared 308/310 collapses, and the selected 0.5 micrometre 308/310 shared-star move. It then rebuilds the existing diaphragm registration and derives row 310 with the pinned pleura owner. No vertices move in the 306 collapse or 307/309 flip; the two 308 collapses remove only their recorded local vertices/faces. The star operation changes one shared coordinate in both owners.

Each reproduction now derives pre-pleura and final respiration-configuration metadata from the exact emitted NHA and its lobe rows. It writes current whole-payload and per-lobe geometry hashes, recomputes the Kuhn-basis area/volume derivation, and rejects a config with stale geometry metadata even when its runtime diaphragm-area scalar matches. Numeric physiology parameters and the runtime Float32 diaphragm area remain unchanged. The frozen v8 output and its receipt are retained as historical inputs; the metadata-only 1116 successor separately binds the same NHA to a current derivation.

The compact bundle includes the composition report, reciprocal-map summary, final manifest, and the exact 303,656-row row310 lineage sidecar. Larger NHA, receipt, and full reciprocal map files remain at their pinned evidence paths; retained-artifacts.json records their hashes. The output records closed oriented lobe shells, closed row 311, and row 310 with Euler characteristic -32 and two components. The final row310 face-lineage sidecar maps all 303,656 faces to source lobe IDs 305-309 and source face IDs. The pre-pleura 1105 checkpoint reports 104,975 synthetic stage-parent entries for row 310. That is historical intermediate bookkeeping, not final lineage: the final recooked row310 sidecar is authoritative, and no synthetic stage-parent identifiers survive it. The retained historical receipt field is documented by row310-lineage-scope-correction.json; it is not rewritten in frozen v8.

Per-lobe and aggregate serialized Float32 area and volume checks match the 1078 parent. Saved-pose checks and native GPU replay are separate from these source composition checks. No clinical, pleural-fluid, parietal-layer, or interlobar-fissure-lining claim is made.

## Reproduce on the pinned Mac mini

The exact 1113 composer and its direct owner imports are pinned in source-pins.json. The import source is clean checkout 32c907cebf131d462fe87f2cbdd86e40e6cc9b6b; the later origin/main lung repair module differs and must not be substituted. Create that checkout at /Users/n/numi-human-final-lung-composition-001 if needed, then run:

    /Users/n/numi-human-prep-venv-20261005/bin/python tools/evidence/native-lung-final-selected-composition-1113/reproduce_to.py /Users/n/numi-human-resting-evidence-20261005/native-lung-final-selected-composition-replay-NEW

The wrapper checks source revision, clean status, composer checksum, and all directly imported owner-module hashes. It refuses to overwrite an output directory. Inputs and outputs remain bound to absolute paths and SHA-256 values in the generated report.

## Focused regression run

    /Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q tests/test_native_lung_final_composition_1113.py

The Mac-mini integration suite runs the exact composer once into a temporary directory and checks the regressions found during composition: 1083 parent retention for paired-union maps, distinct row308 and row310 stage inputs with recomputed normals, preservation of provisional qualification metadata, current respiration metadata bound to the emitted final NHA/per-lobe hashes, rejection of a stale-geometry config with the same area scalar, and final row310 topology/lineage. It also verifies the reproduced serialized NHA and per-lobe Float32 area/volume against frozen v8. On machines without the pinned source/evidence inputs, tests skip rather than substituting different anatomy.

## Frozen v8 references

Evidence path: /Users/n/numi-human-resting-evidence-20261005/native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8

- Composition report SHA-256: f2fd49d0486e20bdca5ea8638215466f4a59ff59d94f1dffa53c8caa5018b460
- Final NHA SHA-256: 1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc
- Receipt SHA-256: 118788f2db039f805cff15c0bff7efa8c62ec68595c997564358d14044ee53e8
- Manifest SHA-256: 0b59c2eb417e339c67ed548b766bbd30b81bec5dedd9de5bea30a7a3c96ddadc
- Reciprocal lobe-face map SHA-256: 8f66e5bf2b929bd343dea5f3868fa6348b24f44706daa4d62388276ddb1d7d56
- Reciprocal lobe-edge map SHA-256: 45060d64f69361d285118dc0ee0ed39ed1502bc6f8d8a2b7bab56eb1659c1089
- Final row310 lineage SHA-256: 2e2c82e96faaafd9f0c6ef3245bc865adeabfcccf3ce5b3a0366c1369fc9d3ba


## Metadata-only successor 1116

The frozen v8 payload is unchanged; the metadata successor binds its respiration configuration, receipt, and manifest to the exact emitted NHA and current rows 305–309. The runtime diaphragm area remains `0.018687047064304352` as Float32, and all numeric/physical respiration fields are unchanged. This metadata refresh does not itself qualify native geometry or a long cycle.

- Report: `/Users/n/numi-human-resting-evidence-20261005/native-lung-final-metadata-refresh-1116/metadata-refresh-report.json` (SHA-256 `bc58b6ab97d352ab1d5a68e9c9277a4b8f8161daf2d3b41a9eaff6ca8429f48c`)
- Respiration config: `/Users/n/numi-human-resting-evidence-20261005/native-lung-final-metadata-refresh-1116/final/resting-reference-respiration.json` (SHA-256 `0e1437022b8c7d831fd9d2c9db6fb6116a9e7b5c9b2569c62b4b95d290ffcaa5`)
- Receipt: SHA-256 `12d86a9295dd2d1f926ee50066a0d46cce8c3bcc2ec401de1824d9e486503c06`
- Manifest: SHA-256 `703c23e948fa4da4f58921678e2d3d6c8c162850e3020094cf96f48a78c61d6d`
