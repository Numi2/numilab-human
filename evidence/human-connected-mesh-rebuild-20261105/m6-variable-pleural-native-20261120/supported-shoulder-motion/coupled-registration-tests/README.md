# Coupled shoulder and vessel registration tests

The arm-base calibration cleared both joint surfaces but the two left vascular rest maps failed. With the existing trunk and arm attachment regions held fixed, material-path lengths set a lower bound on the required stretch. Every original triangular edge along a particular path contributes its original length. The distance between the two moved endpoints cannot exceed the sum of the moved edge lengths. This gives a bound independent of how the intervening vessel is smoothed or rerouted with that material connectivity.

| Passive vessel row | Original endpoint IDs | Original edge path | New endpoint separation | Lower bound on maximum edge stretch |
| --- | --- | --- | --- | --- |
| 475 | 246 and 728 | 82.97 mm | 108.34 mm | 1.3058× |
| 477 | 303 and 868 | 111.49 mm | 136.39 mm | 1.2234× |

These are geometric requirements for the candidate reference with the observed parent assignments. They are neither physiological limits nor measured vessel strain. The proof uses original triangle edges, including their separate seam indices. It restricts endpoints to vertices fully assigned to the trunk or the left arm. The primary source payloads and native rest-pose oracle are listed by the preceding qualification.

A second test fixed the source-derived shoulder joint region and searched three angular degrees of freedom with a distal forearm objective. On the left it also constrained both vascular endpoint separations by the corresponding original material paths. Those necessary constraints could be satisfied, but that candidate still intersected the scapula in all five examined poses. The source-orientation candidate cleared the bones and violated both vascular bounds. Intermediate orientations retained at least one failure. This is a bounded experiment. It leaves open other coupled registrations involving the superior chest, shoulder girdle and material reference.

Holding the arms fixed and applying the inverse calibration on both scapulae introduced a left clavicle crossing. Preserving both clavicular and scapular material lengths can match three source-derived joint landmarks algebraically. The tested two-link fit still intersected the surrounding bones. Preserving landmark distances alone is insufficient for anatomical admission.

The accepted change in this checkpoint is the native enclosed-span check and regression in the parent directory. The alternative registrations are retained as calibration experiments. The integrated scene has no accepted steps with the new shoulder placements.

## Reproduction

Run the parent reproduce.sh first to obtain its extracted poses.csv and comparison outputs. Use the input paths in the parent supported-run.json. For the programs in this directory, add the parent directory and its existing native tools and exact-predicate source directories to PYTHONPATH. Run check_vessel_stretch.py with the same arguments as check_vessel_transfer.py to emit the endpoint witness JSON and NPZ. Run fit_vascular_joint.py with the parent registration inputs, the atlas archive, articular-frame.json and that NPZ. Run girdle_ik.py with the same registration inputs, atlas and articular-frame.json. These programs run at authoring time.
