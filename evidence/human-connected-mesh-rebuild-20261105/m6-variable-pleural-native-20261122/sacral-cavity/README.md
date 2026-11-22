# Conservative retetrahedralization under real bed contact

The native quarter-displacement scene contains 1,448,804 skin and fat elements. Its second 15.625 µs step exhausts the nonlinear solve. Sacral fat element 1067701 originally forces the candidate to the minimum volume ratio of 0.35.

The experimental preparation replaces the three tetrahedra in that complete edge star with two. Twenty surrounding cells keep their ordered vertices and primary anatomical labels. They provide enough local mass capacity to conserve the reference and loaded spatial first moments. The face-connected mass neighborhood and the changed geometric cavity have different boundaries. Only the latter changes topology. The loader verifies the exact oriented geometric boundary, source radius, density, donor ledger and loaded pose. The physical mesh uses the replacement in its material and contact solve.

The traced source contains a 2.71 × 10⁻²⁰ kg contribution. Visiting the sources in ascending mass preserves that contribution in the existing compensated flow solver. A C++ fixture replays all 31 source rows and 22 destinations and also rejects a disconnected source. A second fixture checks physical force against the finite difference of the material energy.

| Measured on the second attempted step | Original | Single repaired cavity |
| --- | ---: | ---: |
| Minimum skin/fat volume ratio | 0.350000024 | 0.377549022 |
| Fresh nonlinear residual | 0.05378294 | 0.04605169 |
| Accepted consecutive steps | 1 | 1 |

The new core has minimum normalized determinant 0.05381 at rest, up from 0.000249. Across matched installed and one-step coordinates its elastic energy differed by 0.18% and 1.77%. The corresponding local nodal-force differences were 0.0351 N and 0.1569 N in L2 norm. These local force differences are substantial relative to the original core forces. The endpoint comparison uses the original accepted coordinates. It does not establish equivalence of trajectories.

The instrumented replay independently reproduced the physical result. In its final Newton iteration original adipose source row 719642 reaches the upper material volume ratio of 2.5. The material trial accepts only 5.96 × 10⁻⁸ of the correction and deformable contact then reduces it to 9.14 × 10⁻⁹. Increasing the linear iteration budget does not address either restriction. The dense candidate remains experimental.

The archive receipt identifies the tested binary, native sources and proposal. The replay command and external dependencies are pinned in the launch record. The build tool applies the source patch to the corresponding Human owner and verifies all changed source hashes before compiling. To replay the bounded native carrier fixture, pass the original or reconstructed owner, the directory of the reconstructed headers, the carrier graph log and the matching local mass fit to `check_native_cavity_flow.py`.
