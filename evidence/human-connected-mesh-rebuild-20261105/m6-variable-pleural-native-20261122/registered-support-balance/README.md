# Physical support and joint recruitment on Apple M6

The same registered Human runtime accepted two 15.625 µs steps in each comparison. Skin and fat use 1,448,799 tetrahedra. The existing MyoSim muscles, compliant tendons, gravity, 34 ligament force rows, 26 fascia regions, closed circulation, Brain controller, variable pleural-pressure configuration and sole physical skin–bed contact owner remained active. Each case starts from zero velocity. Pleural pressure remained constant over this short interval.

| Initialization | Active muscles | Initial maximum generalized acceleration residual (m/s²) | Right / left knee torque residual (N·m) | Step-two bed reaction (N) | Deep-fat self-intersections | Skin–tissue pairs |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Six-cavity reference | 0 | 575.586 | 19.00611 / 16.55975 | 3491.918 | 0 | 11,795 |
| Local fat relaxation | 0 | 575.586 | 19.00611 / 16.55975 | 3491.917 | 0 | 11,795 |
| Local fat relaxation, second proposal | 0 | 575.586 | 19.00611 / 16.55975 | 3491.916 | 0 | 11,795 |
| Redistributed bed reactions | 0 | 575.586 | 19.00611 / 16.55975 | 3492.004 | 0 | 11,795 |
| Muscle recruitment capped at 0.25 | 153 | 67.368 | 0.00010 / 0.00008 | 3491.884 | 6 | 11,791 |
| Muscle recruitment capped at 0.125 | 140 | 51.284 | 29.75781 / 24.75283 | 3491.642 | 0 | 11,791 |

The physical frame, input hashes, checks against 878 anatomical banks, separated ocular pairs, comparison clocks and GitHub recovery receipts accompany these results. The acceleration diagnostic omits the floating root and the true FEM bed reaction. The knee values measure initialization. The physical bed reaction is reported separately from the accepted step.

At step two the six-cavity reference had zero inverted material cells, one adipose shape below 1e−5, and a physical bed reaction of 3491.92 N against 952.86 N of body weight. Its cold throughput was 1.563e-07×. With the 0.25 recruitment limit the two-step cold factor was 1.495e-07×. On the third step the original and recruited trials rolled back with fresh physical residuals 0.0677739 and 0.0675745.

Changing 115 bed-normal initial coordinates lowered the computed stationary vertical residual and preserved total load and moments. In the native run it left all 11,795 intersecting skin–tissue triangle identities in place. The two local interior-node proposals separately reduced static material energy but worsened the accepted fat-cell shape. They remain comparison cases.

At recruitment cap 0.25 the exact deep-fat boundary regression is confined to five triangles and eight nodes on registered donor 23. Those nodes differ from the passive accepted positions by at most 21.25 nm. At cap 0.125 the four reported initial leg torques move differently. This sensitivity makes scalar recruitment a poor tool for repairing anatomical clearance.

Five fat-cell witnesses at the rejected third step have every vertex on the shared dermis–adipose interface. The limiting cell (source 947257) is part of a three-cell face-connected region with four faces adjoining skin and four adjoining other fat. At the recruited second endpoint the region occupied 1.050 times its reference volume while the individual cell occupied 2.362 times its own. At the next Newton iterate that cell reached its volume limit. An exact integer face-and-volume enumeration tested 32 small fixed-boundary patches using the existing vertices and registered material. Only one alternate filling met the geometry tests, in another region. Its installed nodal forces differed by 70%, and its matched endpoint forces by 94%. All proposed replacements failed the 1% force and energy comparison. The search covers the reported patches with up to two same-lineage face neighbors.

The supported reduced control carries 952.81 N averaged over the final two seconds of its 60 s run. Its normal-section skin and fat stiffnesses differ from the dense material. Using the dense law in a laterally confined, uniformly loaded column yields a fat pressure of 17.38 kPa at J = 0.35. Three measured regional mean tractions exceed that value. These inferred column stretches are a test of a proposed reduction and are separate from measured 3-D deformation.

## Integration decision

Keep the supported coupled model for breathing, circulation and performance comparisons and the six-cavity dense state for tissue fidelity. Keep both recruited states out of the resting default. The next registration change belongs on both sides of the shared material interface and must retain the outer envelope, opposite-side contact, measured thickness and mass transfer. Use the retained limiting region, deep-fold triangle IDs and accepted frame as regression inputs. Bed preload, cap tuning and same-side flips have now been evaluated on the actual model. Whole-body settling, anatomical clearance throughout motion, and the dense 16 s performance target remain open.

The existing endpoint verifier now reports anatomical eligibility separately and accepts `--require-clear` for admission. A fresh full geometry audit of the intermediate-recruitment frame reproduced the original measurements and returned failure for the 11,791 physical skin–tissue triangle pairs and the one low-quality fat cell. The experimental fit and retriangulation implementations, their tests and compact results are in the independently restored release archive indexed by `registration-study-source-recovery.json`.
