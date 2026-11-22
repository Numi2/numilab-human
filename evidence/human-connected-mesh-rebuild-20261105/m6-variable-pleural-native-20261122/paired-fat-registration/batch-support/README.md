# Supported skin and fat under load

Four accepted steps are retained for solver regression. Whole-mesh measurements disqualify that registered pose as a resting configuration. For each phase the same 1,448,799 installed tetrahedra were evaluated using the registered reference, fixed cavities and the native dermal and adipose constitutive laws. The full mesh data are in `full-material-strain.json`.

| Phase | Outside the existing 2/3 to 1.5 registration stretch interval | Outside 1/3 to 3 | Largest stretch |
| --- | ---: | ---: | ---: |
| Prior loaded pose, before the last three cell repairs | 90 | 29 | 45.45 |
| Four-step candidate at initialization | 110 | 34 | 45.45 |
| Four-step candidate after 62.5 µs | 160 | 38 | 30.47 |
| Prospective 15-cell fit at initialization | 208 | 52 | 45.45 |

The intervals are engineering diagnostics. The 2/3 to 1.5 interval is enforced in authoring for the supported initial pose. Dynamic trajectories retain the native physical acceptance checks. Material stresses use absolute world components of the first Piola stress tensor. A small volume of tissue can exceed the stretch interval even when whole-body averages are close to one.

The 15-cell fit included the physical bed barrier and covered 1,718 incident tetrahedra. After Float32 emission the total initial bed reaction changed by 0.000038 N. One supported contact unloaded without disturbing the first-contact gap. The old generated loader expected a count of 143 for every registered pose. The updated loader checks the full set of authored skin point IDs against the actual loaded coordinates, including this 142-contact case. Native loading passed and the first physical step then rolled back. The dermal witness reached 7,085,721.5 Pa against the material cap of 5,000,000 Pa. Independent evaluation obtained 7,085,724.6 Pa. The whole-mesh authoring check now rejects the same input before creating an output or compiling it. Its relative volume of 0.603 had passed the previous geometric screen. The archived fit's acceleration scope label predates the addition of the bed barrier. The implementation and the independent support check include that force.

No tissue material coefficients, muscle actuators, contact tolerances or solver thresholds were changed. The 16 s motion requirement and the integrated throughput target remain open. The source, rejected executable and signed-off input hashes can be recovered using the release receipt in this directory.
