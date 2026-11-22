# Bonded skin and fat at release

A native replay on SSH Numi’s Apple M6 cleared the six deep-fat self-intersections in the recruited reference case. Both cases accepted two 15.625 µs steps with the same muscle recruitment, real skin–bed contact and variable-pleural configuration. The repaired case had zero outer-skin self-intersections, zero intersections across the three skin/fat surfaces and zero inverted material cells. The endpoint still had 11,791 physical skin–tissue triangle pairs and one adipose cell below the existing shape threshold. Pleural pressure stayed at −490.332 Pa over this interval.

| Two-step measurement | Recruited reference | Repaired registration |
| --- | ---: | ---: |
| Deep-fat self-intersections | 6 | 0 |
| Skin–tissue crossing pairs | 11,791 | 11,791 |
| Fat cells below the shape threshold | 1 | 1 |
| Skin–bed normal reaction | 3,491.88 N | 3,479.82 N |

The reference weight is 952.86 N. The repaired run simulated 31.25 µs in 209.62 wall seconds including initialization, or 1.49×10⁻⁷ real time. The native horizon alone achieved 4.78×10⁻⁷. The full scene still needs supported settling, verified joint loading, respiratory motion and the 16-second integrated clearance and timing check.

The repair changed the registered initial coordinates of a bonded interface and eight nodes in a separate fat fragment. The native material laws, mapped masses, attachment ownership and solver acceptance gates are unchanged. The two material fits use separate element stars. All deep attachment sites, the physical outer surface and the bed footprint retain their previous coordinates. On the shared patch the initial free-node force norm fell from 32.72 to 13.80 N. The skin and fat contributed forces to the same interface degrees of freedom. At the third step the shared-interface trial reduced the fresh residual from 0.06757 to 0.05252 and rolled back at the lower adipose volume bound. A wider 48-node registration also rolled back, with a fresh residual of 0.06549. The initial elastic load decomposition identifies the dermis as the dominant contributor to volume acceleration at the five limiting fat cells.

The installed material mesh contains 89 connected components. The main component holds all 14 direct entheses. The other 88 components contain 1,291 adipose tetrahedra on donor 23 and weigh 0.377 mg in total. The six deep-fat intersections occurred in one 17-node component weighing 2.08 µg. Exact point-in-tetrahedron tests placed every detached vertex and cell barycenter outside the main dermis and fat solids. All 66 affected mass groups also have material in the main component. Those group identities preserve a starting point for a source-level repair. They establish neither a physical attachment nor a conservative replacement. All five volume-limit witnesses belong to the main component.

The accepted checkpoint and native executable are independently recoverable from the GitHub release assets identified by the receipts here. The failed trials and authored poses are retained separately. The native comparison contains hash-bound results and timings. This checkpoint is experimental.

## Three accepted steps with a force-balanced interface

A 48-node registration minimized the initial acceleration of volume change at the five rejecting adipose cells. Both dermis and adipose forces entered through the actual constitutive law and assigned nodal masses. The largest calculated elastic contribution, scaled by the square of one native time increment, fell from 0.535 to 0.00511. This is an initialization calculation. The full native solver subsequently accepted its previously rejected third step with a line-search fraction of 1.

Rounding the first proposal to binary32 produced a first-moment error of 1.64×10⁻¹³ kg·m against a permitted 1.56×10⁻¹³ kg·m. The revised emitter balanced nearby representable coordinates. The native mass gate passed with a maximum absolute first-moment error of 1.78×10⁻¹⁵ kg·m. Offline replay classified all four retained mass-admission cases in agreement with the native owner.

The full scene accepted 46.875 µs in 228.63 wall seconds, or 2.05×10⁻⁷ real time including initialization. The third-step endpoint had zero inverted cells, zero outer-skin self-intersections, two low-quality fat cells, 11 deep-fat crossing pairs and 11,790 physical skin–tissue pairs. Six deep-fat pairs are in the 17-node detached component and five are in the main body. Both low-quality cells are in the main body. The bed reaction remained 2,988.68 N against 952.86 N of body weight. The pulse and breathing response cannot be qualified on this time interval. The 16-second target and sustained operating speed remain open.

The current dermal stiffness draws on forearm measurements at 0.2–1 kHz. Replacing it with a softer scalar to gain solver acceptance would require new calibration. Loaded gluteal MRI provides an independent type of measurement for testing a reduced support model against tissue deformation. [Forearm elastography](https://www.sciencedirect.com/science/article/pii/S1742706122002392) · [Loaded gluteal MRI](https://pubmed.ncbi.nlm.nih.gov/20051619/)
