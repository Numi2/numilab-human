# Ankle recruitment and FEM bed load on Apple M6

The full scene still fails the supported-rest gate. The flat-bed recruitment preset reduces the ankle release transient and passes a short physical prefix.

| Check | Measured result |
| --- | --- |
| Bilateral ankle release | Existing MyoSim recruitment with a 0.10 activation cap changed first-step right/left accelerations from −580.58/−572.33 to −11.91/−14.76 rad/s². The four-step flat-bed run accepted all roots. Knees became the largest joint accelerations at +198.57/+151.85 rad/s². |
| Real FEM bed | The existing plane was pitched 0.144° for this diagnostic and started with 9.10 µm minimum outer-skin clearance. Matter’s vertical reaction was 18.68 N on the first root and zero thereafter, against 952.86 N of body weight. All eight physical roots accepted. The settlement gate failed with a minimum gap of 10.09 µm and max FEM speed of 3.3855 m/s at 125 µs. |
| Active solves | With the same recruitment, the first four tilted roots used 1,840 Krylov columns versus 2,112 on the flat bed. Native command time was 88.40 s versus 101.47 s. This compares two initial contact configurations. The eight-step tilted run spent 176.29 s in native commands, or 7.09e-7× at 15.625 µs per root. |
| Solver experiments | The half-width gather passed the isolated bitwise fixture but the rebuilt full-scene candidate took more columns and failed 158 force and 222 strain comparisons. Its whole-body comparison used the retained library as control, so the whole-binary difference cannot be attributed solely to the gather. The exact graph product variant matched outputs byte for byte across 8,192 synthetic patches and was 1.13% slower by median GPU command duration. Both experiments remain off. |

The bed uses the original Matter contact law and outer-skin rows. The legacy articulated geometry query retains the upward-Z selector for source observation and seeds zero force. The settlement observer measures signed gaps against the compiled plane and compares vertical reaction to weight. Gravity, attachment laws, tolerances, accepted-state checks and anatomy remain unchanged. The tested placement and observer patches are retained as an experiment. No production scene defaults changed.

Pinned launch plans reproduce the two initialized states on SSH Numi from the preserved checkpoint. The result records source and binary identity, solver work, native timings, acceptance and support observations. The remaining physical prerequisite is a deformable equilibrium under real support, gravity and vascular loading before advancing the coupled clock. This prefix gives no qualification through visible motion or repeated physiological cycles.
