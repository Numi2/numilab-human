# Registered-pose material release on Apple M6

On SSH Numi, run `~/Work/run-numi-human-rest --registered-material` for the reproducible 79-step prefix or `~/Work/run-numi-human-rest --registered-material-16s` for the full request. Both use the existing 2 ms native scene. Earlier modes remain available through `--help`.

The previous two shoulder angles placed the dense loaded tissue in a different skeletal pose. Mapping that tissue to the old pose in Float32 produced 2,115 inverted cells. The native skeleton now starts at the registered shoulder angles. The installed 1,448,799-cell volume has positive determinants in both the reference and loaded opening states. A conservative mapping accounts for 3.452571 kg dermis and 10.487108 kg fat.

A ray from each bed-loaded source point follows actual shared tetrahedral faces. The force owner uses connected material thickness at 3,564 of 49,209 source points. The rest retain the earlier nearest thickness assignment. All updated material rows match the installed connectivity and the native solver's emitted Float32 values. This is a reduced normal contact model. The moving scene integrates the articulated body. The full dense skin/fat volume was checked at registration.

At the same registered pose, the control rejected after 146 ms, connected thickness after 150 ms, and recalibrated bed loading with load-aware recruitment after 158 ms. Two independent launches produced the same terminal root and physiology. The 16 s request rejected at trial 80, at a shoulder compression 45.9 micrometres past the material limit. The two cold prefix runs achieved 0.0103–0.0105 times real time. Sustained speed remains unqualified.

At 158 ms the endpoint audit found 8,348 nonocular triangle-pair crossings and 27 skin self-intersections. Native sampled root assistance was zero. Whole-body equilibrium, crossing-free motion and complete cardiopulmonary cycles remain open. The recovery manifest identifies the source inputs, executable, Metal libraries, accepted geometry and failed trials by digest. Every archived member was verified by independent GitHub download.
