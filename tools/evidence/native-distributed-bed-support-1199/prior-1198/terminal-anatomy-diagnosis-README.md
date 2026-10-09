# Terminal anatomy diagnosis: 1198 versus 1191

This is a read-only diagnosis of the single accepted terminal capture from fixed-bed run 1198. It compares the accepted registered body transforms at step 0 and step 10000 against run 1191, then inventories every exact skin-to-target intersection at 1198 step 10000. It is not a full-cycle anatomy qualification or a causal experiment.

Reproduce with the pinned Mac mini interpreter:

    /Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/contoured-bed-reference-1196/terminal-anatomy-diagnosis-1198-vs-1191-001/diagnose_terminal_pose.py

The report is diagnosis.json (SHA-256 3b4739b83fecd3d8c2c513ca04b42775bae13efe9a93f5874ef46916cace1e21). The executable analysis source is diagnose_terminal_pose.py (SHA-256 324efef3f4e6f861fa342d054bc15ad29062c279c92c8782e06a39cda39e96d8). The report lists hashes for all consumed receipts, exact target maps/witnesses, source manifests, and the 1191 baseline clearance evidence.

The 1198 accepted start has the same body-state hash, respiratory-state hash, and 86 registered body poses as 1191 step 0. At the terminal, 1198 has 3,727 exact skin-to-target triangle-pair intersections across 26 targets: pelvis surfaces (30), bilateral lower-limb muscle surfaces (2,490), bilateral posterior tibial veins (471), right calcaneal tendon (42), and left wrist/hand muscle surfaces (694). All 859 target pairs and the skin-self pair were covered at this one capture; the 17 ocular targets had zero intersections. The 1191 baseline is supported by the complete 937 target scan and 1187 changed-skin differential across the declared eight captures; those are separate, already-pinned evidence.

Accepted segment transforms show a changed articulated pose, especially at the hips and upper limbs. Pelvis-to-femur relative orientation differs between the two terminal captures by 26.56 degrees on the right and 26.09 degrees on the left; knee relative orientation differs by only about 0.0017 degrees right and 0.000008 degrees left. These are three-dimensional relative segment rotations derived from registered body poses. Run 1198 did not emit scalar generalized-coordinate diagnostics, so these measurements cannot be compared to the separate scalar hip-fit interval [-12, 0]. They also do not isolate the fixed bed as the sole cause of the pose change.

Only the terminal step was scanned in 1198. The scan therefore establishes a terminal failure, not the timing or onset of crossings over the run. The evidence does not justify a full-horizon anatomical claim.
