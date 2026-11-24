# Left forearm load and material registration on Apple M6

In the retained control, the unconstrained second-root correction takes fat cell 1433491 from a candidate volume ratio of 2.49999 to 4.02542. The validity limit is 2.5. In the adjacent skin cell 1394217, the starting ratio is 0.46005 and its contribution along the peak nodal force is 484 N. These values come from paired GPU displacement words and independent evaluation of the registered material law.

Six old fat tetrahedra can be replaced with five while leaving surface coordinates fixed. The proposed faces lose their inherited source-face labels on P1 replay. Re-lumping the geometric mass transfer changes a nodal mass by up to 0.288 g. Its isolated nonaffine force distribution also changes. This topology was rejected.

The registration trials retain all source nodes, elastic cells, source mass, muscle and ligament bindings. They initialize selected coordinates of the bonded internal interface. The six fitted interface nodes move under the original operators during execution. The four-node and six-node energy fits introduce excessive shear despite valid determinants. The strain-bounded candidate protects both principal stretches in every incident cell in the initial pose and under the retained control displacement. The latter check is a replay prediction. Actual accepted time is reported below.

| Full-scene trial | Accepted steps | Simulated seconds | Second-root mechanical residual | Active columns in second root | Launch seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1 | 7.81250037e-06 | 0.00152729778 | 393 | 202.03 |
| four node | 1 | 7.81250037e-06 | 0.000271082448 | 561 | 225.90 |
| six node | 1 | 7.81250037e-06 | 0.000102487385 | 732 | 213.27 |
| strain bounded | 1 | 7.81250037e-06 | 0.000102514168 | 738 | 212.26 |
| bounded one step | 1 | 7.81250037e-06 | unavailable | unavailable | 177.66 |
| half step | 2 | 7.81250037e-06 | 2.71437384e-05 | 456 | 234.15 |

All trials used the variable pleural pressure setting and full bonded tissue mechanics. Native rejections remain in the record. Skin/fat wall and cortical comparisons in the edited region do not qualify the remaining body. The whole-body 16 s motion and throughput gates remain outstanding. The recovery archive contains the launch configurations, code, source hashes, registered coordinates, native binary for the strain-bounded candidate, and full comparison traces.

The accepted one-step capture has 0 surface intersections across the four material walls. It has 4 skin and 26 fat cells below the shape threshold. The cortical solid test across 185 bones gives 5.976699 cubic centimeters of summed pairwise intersection.
The matched mechanical capture passes the stretch comparison on all 144 directly affected cells. Across the rest of the body 2 skin/fat cells exceed the diagnostic 0.5 percent envelope outside the near-identity window.
Across the full shell the number of nodes with more than 100 N of internal elastic resultant changes from 64 to 62. External loads and inertia enter separately. The mass map and original physical support remain active.
After the registration, the peak final certified residual is in object 15. The skin/fat peak is at zero-based node 110041. Raw GPU measurements reproduce the fresh residual within relative error 2.75e-09.

The finer clock was a matched-time diagnostic at 3.90625e-06 s per step. It accepted 2 steps and reached 7.81250037e-06 physical seconds. Compare physical time and fresh certificates when the step length differs.
A subsequent local energy fit at the right-ulna witness moved the initially high fat volume ratio from 2.4900 to 2.4925. Its limited change in local initial forces did not justify another native registration trial. The measured starting force at that node was 1.55 N. Its terminal residual includes inertia and interactions. Further right-side work requires the actual late-motion force and material direction.
