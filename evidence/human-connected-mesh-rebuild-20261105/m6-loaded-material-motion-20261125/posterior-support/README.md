# Posterior load transfer on SSH Numi

Run the matched full-volume GPU release with `scripts/run_resting_human_on_numi.sh --dense --posterior-support pressure-fit`. Add `--check` to verify the inputs. The dense control and this release use the same two steps of 7.8125 µs and the same registered bed, tissue material and variable pleural pressure.

At the opening pose the numerical collision barrier in Metal produced 120,393 N at zero velocity. A smooth inward displacement of at most 3.503 µm at 325 skin/fat nodes brought this frozen-pose force to 952.85 N. The normal component of weight was 952.86 N. Most pressure still acted over one small gluteal patch, leaving about 86 N·m of moment. In the coupled release the terminal mean bed reaction rose from 10.991 N to 11.012 N. Peak recoil at the ten initial contact sites fell from 19.8 to 13.0 µm. The full release took 219.59 s. The material audit at the terminal frame still found 563 stretch failures, 37 poor cell shapes, four relative-volume failures and 908 cortical occupancy samples. Zero cells inverted. This state does not meet supported rest or the motion qualification.

The three observed contact sites would need approximately 171, 241 and 541 N to balance weight and moment at their current positions. The modeled deep gluteal compartment covers only part of the load path to the pelvis and femurs. Its constitutive stress and allowed strain bound the total direct force on its fat parents to 30.99 N. Adjacent whole faces of the registered fat and bilateral gluteus surfaces extend that compartment from 31 to 67 faces, with 76.20 cm² of projected area. Ten incomplete source faces and two intersecting existing fat were excluded. The same material then admits a 77.52 N upper bound. These are ceilings for this compartment, not observed support reactions. The expanded donor ledger accounts for 95.43 g within the existing rigid mass ownership. The expanded GPU source, closed-region certificates and force tests are recoverable from the build release.

The archives list each member with its SHA-256 after independent download and recovery. The tissue assembly uses registered mixed-source reference anatomy. Two historical visible muscle surface overlaps remain in the expanded volume (0.0441 and 0.00246 mm³). The original registered skin and fat showed zero positive-volume overlaps with the retained expansion in the two evaluated frames.

## Expanded native release

Replay the 67-face connection with `scripts/run_resting_human_on_numi.sh --dense --posterior-support expanded`. The 291-coordinate Metal operator transfers the deep-fat forces to the pelvis and both femurs. The 51-node and 91-node specializations share one shader. The GPU processes rows above 255 by striding across the workgroup. The test compares scattered forces and torques with independent virtual-work derivatives for rest, imposed compression, rigid rotation and combined motion.

The two accepted steps took 219.29 s including preparation and presentation, for 15.625 µs of simulated motion. Both variable-pleural launches retained the same terminal bed reaction of 11.0117 N. The expanded interface transferred 0.000569 N to the skeletal donors along the bed normal on the last observed step, compared with 0.000344 N for the 31-face interface. The independent nonlinear local check measured determinant ratios from 0.999696 to 1.000138 in the expanded tissue. The whole scene still fails the skin/fat and cortical occupancy audit. The reported root acceleration residual at release was 68.31 m/s². This run does not qualify 16 s of supported motion.

A separate fixed-geometry Metal check shows that halving the timestep multiplies the opening collision-barrier force by four. Matching that force to weight gives a timestep-specific preload. It cannot establish equilibrium. A static force and moment calculation over all the clipped opening triangles requires at least 499.75 N on the largest patch, even when the pressure may move anywhere on those triangles. The three patches total only 1.11 cm² of projected skin. The area-by-depth screen measures how much of the real posterior surface is near the fixed bed. It changes no positions and applies no forces.

The next physical initialization needs broad contact formed through tissue deformation and a load path that balances the skeleton under gravity. Judge it by accepted support impulses, center of pressure, transmitted skeletal load, residual acceleration and tissue validity together. Force-matching the collision barrier and enlarging the same small compartment both failed to produce supported rest.

### Right teres major in the native release

The recovered BodyParts3D right teres major (FJ1507, stable ID 87) uses its original oriented triangle chain and its authored humeral and scapular weights. Its registration follows the physical deep-fat boundary and preserves the selected tendon-adjacent source vertices. Restoring the source replaces a previously reconstructed surface with a different enclosed volume. MyoSim force routes, both named tendon sites, physical mass tables, gluteal coupling, variable pleural pressure and the other 163 tissue banks use the same source as the control. The two source and motion recovery receipts identify the original atlas attribution, registration, commands, native assets and captures.

The matched native endpoints at 15.625 µs gave the following triangle-pair counts for this muscle.

| Intersection | Control | Rebuilt |
|---|---:|---:|
| Nonadjacent faces of the muscle | 139 | 0 |
| Outer skin | 50 | 0 |
| Skin–fat interface | 44 | 0 |
| Deep fat boundary | 49 | 0 |
| Neighboring anatomical banks | 14,337 | 590 |

Twelve neighboring banks intersected the reconstructed control. Five still intersect the rebuilt muscle. Their individual counts are in `covered-shoulder-geometry.json`. The mesh topology changed, so face-pair counts alone cannot measure penetration volume. The original source repair had self-intersections, which invalidate an occupied-volume interpretation of its Boolean estimates. The rebuilt surface is also farther from the declared scapular route endpoint. The physical span from muscle to tendon needs separate registration.

The expanded control and the rebuilt scene each transferred 11.012 N through the bed and approximately 0.000569 N through the observed subgluteal interface at the final root. The gravitational load is 952.864 N. Both retained 37 low-shape cells, 563 stretch failures and four deformation-volume failures. The rebuilt release took 218.80 s end to end for two accepted physical steps on SSH Numi. It demonstrates a local anatomical correction. Sustained resting performance, integrated physiological cycles and whole-body contact still require qualification.

Run the published scene on Numi with `bash scripts/run_resting_human_on_numi.sh --dense --posterior-support covered-shoulder --steps 2`. Use `--check` in place of `--steps 2` to verify the replay inputs. The public manifest adds the three cardiac sidecar hashes to the exact measured invocation. To reproduce the tissue bytes separately, run `rebuild_covered_teres.py` with the expanded parent run, original NHTISS donor, FJ1507 OBJ, retained deep-fat registration proposal, source receipt and a fresh output directory.

The independent starting-state experiment used the registered skin and fat material laws, their actual masses, contact and the gluteal stiffness matrix. Fifteen disconnected pieces in the 40 mm posterior trial have a free translation. Treating their uniform gravity in a freely accelerating center-of-mass frame removed the false static convergence. The reduced full-weight equilibrium then balanced its chosen force and moments but introduced 312 principal-stretch and 32 deformation-volume failures. A separate strain-limited release reduced the terminal bed force to 10.608 N. Both failed predictors are retained in the source recovery archive. The coupled scene continues to use the measured control starting field.
