# Scalp skin/fat connectivity

This authoring candidate changes eight disjoint scalp quadrilaterals in the current 7,233-node bonded shell. Each edit updates the outer skin, bonded skin/fat interface and deep fat, replacing twelve tetrahedral rows while retaining every node coordinate, thickness, anatomical bank, and cavity perimeter. None of the changed faces is in the bed footprint.

| Whole-shell skin/fat-to-bone audit | Retained native topology | Edited topology evaluated at the same pose |
| --- | ---: | ---: |
| Startup | 17,850 | 17,525 |
| Accepted root 128, 2 ms | 17,897 | 17,572 |

The independent whole-shell audit re-evaluated all 469,136 physical bone triangles. It found no new intersections within or between shell layers at the two captures. Unchanged faces retained the exact same crossing witnesses. The changed faces remove 569 and add 244 triangle-pair witnesses at each capture, with a net improvement of 325 and no increase for any layer/body/bone bank. The verification does not count these as distinct deep penetrations.

All 28,632 tetrahedra have positive orientation and satisfy the existing normalized determinant floor of 1e-5 at the material reference and both captures. The lowest observed normalized determinant is 2.6110654e-5. Source mass closure independently checks all 628 typed pools against the exact source of the preceding mass map. The total is 13.939679838742224 kg, with a maximum error of 8.68e-19 kg for any original source row. The longest newly affected source-to-target route is 15.324 mm. All routes remain within 75 mm, and the existing 900–1,100 kg/m³ dermis and 850–950 kg/m³ adipose bounds hold to the existing numerical tolerance.

The strict geometric-overlap mass screen yielded two eligible double-layer edits and one disjoint edit resolving two pairs. Bounded local transport yielded 13 eligible edits and eight disjoint edits resolving 325. The surface-only screen predicted 716 and 713 removals for 18 disjoint edits. Those predictions omit the volume construction and are not the verified result.

## Native integration

These boundary edits are authoring evidence at fixed native poses. The existing explicit refinement compiler requires each old parent volume to close within 1e-6 and requires exact RegisteredSurface triangles at layer facets. All 16 edited material cavities have unequal old/new reference volumes beyond that tolerance, with a maximum change of 1.747%. The new reference contains about 6.39 cm³ outside the old reference while dropping about 6.22 cm³ of the old domain. A volume-overlap map with exact old/new closure therefore cannot describe this edit. The native registered-volume replacement owner supports a separately identified derived reference. Its integration needs the replacement boundary/source-face ancestry and exact runtime mass and force verification.

The current native result remains the four-cell fixed-boundary replay in the adjacent preload-cavity evidence. It accepted 128 roots over 2 ms with variable pleural pressure. All 34 registered ligament rows passed the short run and five audited shoulder samples were clear. The bed reaction at the endpoint was 0.438 N for a 952.864 N body weight. Body-only throughput was 0.0000204×. Full support equilibrium, continuous crossing clearance and 16 seconds at 0.1× remain open.

## Reproduction

Use the scripts from their original authoring directory on SSH Numi and the retained virtual environment. Let `W` be `~/Work/human-integrated-gaps-20261104`, `T` be `$W/coarse-hand-cavity-024`, and `R` be `$T/preload-cavity-topology-051/native-root-128`.

```sh
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$T:$T/posterior-load-interface-040:$T/posterior-load-path-042:$T/preload-cavity-topology-051:$W/resting-clearance-and-atrium-006:$HOME/Work/skin-fat-native-integration-209/tools"
PY=$HOME/numi-human-prep-venv-20261005/bin/python3
B=$T/thickness-rebuild-034/inputs/physical-bone-initial.bin
M=$T/preload-cavity-topology-051/remap-01/balanced-mass.npz
"$PY" screen.py "$R" "$B" replay-screen.json
"$PY" volume.py "$R" replay-screen.json "$M" replay-strict
"$PY" volume.py "$R" replay-screen.json "$M" replay-transport --bounded-mass-transport
"$PY" verify.py "$R" "$B" "$T/posterior-gluteal-continuum-041/native-candidate-128" "$T" replay-transport replay-exact
"$PY" check_registration.py "$R/shell-mass.csv.nodes.csv" replay-transport/geometry.json replay-exact/registration.json
```
