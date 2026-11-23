# Physical registration of the resting adult on SSH Numi

The rebuilt physical left forearm removes 634 outer skin to anatomy triangle pairs at the reference pose. A full audit of the actual exported mesh tested 152,261 outer triangles against 878 registered native anatomy primitives. Its remaining 10,130 pairs matched the regional reconstruction down to each triangle identity. The release of the preceding full-volume simulation already contained 10,754 of the 10,761 pairs measured after four accepted steps. The repair is a native-importable material candidate. Full-scene integration and motion qualification remain open.

## What passed

| Check on the reconstructed adult | Result at release |
|---|---:|
| Occupied skin and fat nodes | 311,361 |
| Typed skin and fat tetrahedra | 1,450,187 |
| Inverted, undersized, below shape threshold, outside density bounds | 0 in each category |
| Intersections within or between the three skin and fat material surfaces | 0 |
| Physical skin against bone | 0 |
| Registered material facets recovered on exactly one same-material incident cell | 833,108 of 833,108 |
| Dermis and fat mass from the production C++ importer | 3.452571283 kg and 10.487108427 kg |
| Difference between imported and independently calculated nodal masses | 0 kg |
| Left brachioradialis against skin | 746 pairs originally, 61 after repair |
| Right brachioradialis against skin | 613 pairs originally and after repair |

The compiler recomputed the local, piecewise-linear same-material interpolation from the emitted coordinates. Every affected source projection was within the existing 20 mm registration limit. In the two edited regions, 57,352 positive contributions route recorded source mass to 52,001 receiving cells. Current carrier positions, typed region neighborhoods, permitted travel distance and individual source budgets were checked. The longest transfers were 186.723 mm for dermis and 185.510 mm for fat under the inherited 187.444 mm cap. These distances describe numerical mass attribution in the mixed-source reference assembly. They do not measure a biological displacement. Maximum absolute mass and first-moment errors in the edited materials were 2.8e-17 kg and 4.6e-18 kg m. The three source layers, original node identities at registered surfaces and the 14 deep attachments are preserved. The minimum cell quality and volume lie very close to the existing floors. A motion check remains necessary.

## Physical force paths

I tested all 416 muscle actuators in the native NHMYO2 and NHTEND3 inputs at the captured release pose. Of 1,586 named route-site occurrences, 1,301 were inside the skin, 284 were outside and one had ambiguous enclosure. The exceptions comprise 42 origins, 66 insertions and 176 intermediate sites. The forearm repair did not change a site's classification. This point test leaves the 262 wrapping records and the path segments joining sites unqualified.

Of the 108 exterior endpoint centers, 95 actually use the original point attachment. For 91, the compiler recorded that the bone surface exceeded its distance gate. For four, the body had no registered bone surface. The other 13 centers use four-node force distribution. Every one of the 1,464 traction nodes for the body's 366 distributed attachments was inside the skin, including those 13 patches. Three independent unit loads applied to each stored force map gave maximum absolute resultant and moment errors of 5.97e-8 N and 5.80e-9 N m. Enclosure and static force transmission establish neither anatomical insertion accuracy nor motion fidelity.

The bilateral posterior deltoid origin is a specific force-registration failure. Its unregistered point sits 21.24 mm outside the skin on the left and 18.64 mm on the right. A left deltoid intermediate site is 26.63 mm outside. Distances are to the actual imported physical skin. Several latissimus route points are also more than 20 mm outside. Their source actuator, body, local route identity and world witness are recorded. The largest remaining passive surface defect is the right extensor digitorum with 916 triangle pairs. An exterior passive envelope and an exterior mechanical force point are different measurements.

Distinguishing the remaining point attachments from surface-placement errors changes the repair. I checked the 95 exterior point-mode endpoints against every bone surface owned by their native rigid body. Four have no registered cortical surface. None of the other 91 is within the existing 12 mm generic registration threshold. Only two are within the 25 mm maximum for endpoint migration, the bilateral LTpT_T12 insertions at 23.81 mm. The left and right posterior deltoid origins lie 48.75 and 48.77 mm from their respective scapular surfaces. A wider gate would permit a major change to their moment arms. These are distances to actual native geometry. The anatomically intended correction still requires landmark correspondence between the source skeleton, named muscle attachment and target bone. [Individual point-to-cortex witnesses](forearm-unbound-cortex.json)

## Integration decision

Carry forward the left material repair and the existing production solver. Reject the right-side skin trial that increased brachioradialis crossings and the bed-mode solver experiment that increased body runtime from 42.86 to 54.21 seconds with the same 491 Krylov columns. The original full-volume trial advanced 62.5 microseconds. Its final reported bed reaction was about 15 microNewtons for 953 N of body weight. It does not demonstrate supported rest.

Resolve the registration between bone landmarks, force sites and tissue with the deepest bilateral deltoid failures first. Preserve source anatomy, audit signed moment arms and check neighboring tissue against the same native pose. Do not move the 13 already interior traction patches based on exterior route centers alone. Finish the complete surface-node and mass-registration inputs for the rebuilt mesh, then verify gravity transfers through the bed, tissue and skeleton. At present 10,130 intersections across 76 anatomy primitives remain. Sixteen seconds at 0.1 times real time with this full material and variable pleural pressure remains unverified.

[Native import](forearm-current-import.json) · [Source mass routes](forearm-current-carriers.json) · [Boundary facets](forearm-current-facets.json) · [Whole-body audit](forearm-exported-clearance.json) · [Triangle identities](forearm-exported-crossings.jsonl.gz) · [Remaining anatomy](forearm-current-anatomy.json) · [Muscle and tendon findings](forearm-current-force-sites.json) · [Independently verified recovery](forearm-current-recovery.json)
