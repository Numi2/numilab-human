# Sixty seconds of supported native motion

The Apple M6 native viewer completed 30,000 accepted 2 ms steps with the qualified gluteal registration, existing 32 bed sections, active respiratory controller, circulating blood and variable pleural pressure. The cold launch took 574.523 wall seconds for 60.000 simulated seconds, or 0.10443× end to end. Motion after startup measured 0.10682×. The viewer issued 940 presentation callbacks, averaging 1.64 per wall-clock second.

All 77 columns on all 938 coupled observations match the separately retained 60-second control exactly. The model completed 11 breaths and 70 filling/ejection cycles. Pleural pressure covered -770.0 to -470.3 Pa. Late mean bed reaction was 952.81 N with peak root assistance of 0 N and 0 N·m. The terminal blood-gas state was 102.14 mmHg arterial O₂ and 39.37 mmHg arterial CO₂.

There were 938 accepted donor poses plus the independently decoded native entrance and endpoint packs. The 688 observations after second 16 were outside the authoring interval. The largest late count of bilateral and self-intersection pairs in the two gluteal muscles was 0. The late minimum relative cell volume was 0.2983. The corresponding minimum normalized cell determinant was 2.58576e-05. The separate, uncalibrated passive specimen check recorded 0 cell-sample violations. The retained compressed trace contains all three donor bodies for each reported pose. The hash of the complete 157-body diagnostic is in `qualification.json`.

## Full-volume pelvic interface diagnosis

The original BodyParts3D v4 atlas and current native registration both contain muscle-to-bone intersections. The exact pair counts below use the same Float32 predicate. Each source member and registered bone has a checked identity.

| Bone | Right glute, original atlas | Right glute, native entrance | Left glute, original atlas | Left glute, native entrance |
| --- | ---: | ---: | ---: | ---: |
| Sacrum | 372 | 370 | 349 | 372 |
| Right femur | 92 | 81 | 0 | 0 |
| Left femur | 0 | 0 | 90 | 83 |
| Right pelvic bone | 165 | 140 | 0 | 0 |
| Left pelvic bone | 0 | 0 | 161 | 118 |

All twenty bone/muscle/capture comparisons reproduce the independent native pack counts at entrance and step eight. Each of the five exact-coordinate welded bone surfaces is closed, consistently oriented and free of nonadjacent self-intersections. Exact solid point location located muscle-cell centers at least 2.09 mm and 2.18 mm inside the registered sacral surface on the right and left. Disjoint interior balls establish conservative lower bounds of 0.0470 mL and 0.0295 mL of material overlap. The actual shared volumes may be larger. The full-resolution body still needs anatomically conforming attachment boundaries and verified load transfer through the posterior tissues.

The supported run does not contain the complete volumetric skin/fat and ligament model. Its successful motion qualification applies to the bilateral muscle registration and the measured coupled outputs. Whole-body clearance, 15 wall-clock presentation frames per second, the 330-second baseline and the matched respiratory-drive intervention remain open. Source and licensing details are in the [parent registration result](../README.md).
