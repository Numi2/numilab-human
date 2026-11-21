# Connected skin and fat on the registered skeleton

This candidate repairs the physical reference volume for both arms and the shoulder girdle. It uses the same 311,020 nodes, 1,448,804 tetrahedra, material identities, and shared skin/fat boundary as the preceding Human reference. The complete native scene still uses its prior physical assets. This checkpoint covers one captured entrance pose.

| Physical check | Previous geodesic transfer | Saved candidate |
| --- | ---: | ---: |
| Inverted tetrahedra, skin / fat | 1,948 / 1,831 | 0 / 0 |
| Outer skin × bone triangle pairs | 63 | 0 |
| Inner skin × bone triangle pairs | 325 | 0 |
| Deep fat × bone triangle pairs | 2,511 | 1,347 |

The intersection tests cover exactly the 62 moved skeletal banks. All tetrahedra pass the existing signed reference-volume and scaled-determinant limits after Float32 emission. The independent C++ mass-preflight reader evaluates the emitted TetGen nodes against the original element file. The candidate changes total reference volume by +0.18% for skin and +0.65% for fat. The lowest outer node remains 0.0000404 mm above world z=0. This is a coordinate measurement. Support force requires an accepted physical step.

The anatomical transfer integrates the rotations of the four moved bone groups and the surrounding fixed material over their shared volume. The 8 mm screening length is an authoring parameter. The integrated displacement is corrected against the actual material cells, including rounding of the world coordinates to Float32. Subsequent changes fit material boundary contact, signed cell volume and receiver capacity. The same nodal coordinate moves both materials at every shared interface. The native scene does not run these authoring optimizers per step.

The carrier solver conserves 13.93967896346 kg across the 71,639 original carrier quotas. Of 57,137 receiving regions, 53 fall outside their existing local density bounds. Their combined violation is 7.252 g. One skin region has 3.42 times its source volume and accounts for 2.084 g of the shortfall. Global mass conservation therefore cannot qualify this transfer. The graph includes previously dormant original routes. Their spatial and P1 eligibility still needs validation on the updated geometry. Per-cell and nodal mass must be regenerated after the actual bounds close.

The preceding full native launch rejected both shoulder ligament rows before the first physical step. The revised skeletal reference also retains a right shoulder route failure under prescribed rotation. This candidate has no native throughput or motion result. Remaining work includes the deep-fat interfaces, local density and source correspondence, shoulder contact, whole-body collision checks and joint qualification under bed and respiratory loads.

Run `./reproduce.sh` from this directory on SSH Numi to recover the standard TetGen node file, test emitted-cell repair, recompute each volume with the native owner and reproduce every changed-bank crossing and carrier deficit. The script requires the original retained scene, connectivity and two native checkers. The compressed node file and the bitwise recovery checks avoid having to repeat exploratory fits. The receipts pin the pose, original assets, native binaries, node file, volumes and intersections.

BodyParts3D © The Database Center for Life Science, CC Attribution–Share Alike 2.1 Japan. This derivative uses the existing attributed Numi Human reference assembly and its registered MyoSim skeletal frames. It represents mixed-source reference anatomy.
