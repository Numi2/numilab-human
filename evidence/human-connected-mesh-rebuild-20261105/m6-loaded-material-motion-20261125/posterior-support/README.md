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

## Bilateral scapular tendon registration

The teres major scapular endpoints were 56.942 mm from the right BodyParts3D scapula and 56.865 mm from the left in the accepted pose. Both sites were 0.541 mm from the corresponding original MyoSim cortex. Independent source sites reproduced the two native force frames within 4 nm. The assembly mixed different scapular registrations. A direct replacement with the MyoSim bones extended both scapulae below the bed. Unconstrained alignment moved the shoulder joint centres by about 42 mm.

The registered scene keeps both skeletal joint centres, the existing skeletal transforms and the compliant muscle parameters. Each attachment uses four connected points on the registered scapular cortex. Moving the private terminal segment onto the bone initially shortened the entire wrapped path by 14.2% on the right and 14.4% on the left. The corrected via point preserves the complete wrapped length and incoming force direction at release. Only two private via points and two cortical endpoints change in the MyoSim route tables.

| Matched check | Original release | First cortical trial | Registered route |
| --- | ---: | ---: | ---: |
| Right cortical offset at the accepted frame, mm | 56.942 | 0.000051 | 0.000051 |
| Left cortical offset at the accepted frame, mm | 56.865 | 0.000037 | 0.000037 |
| Initial generalized muscle load coordinates exceeding the comparison tolerance, of 128 | 0 | 45 | 0 |
| Complete native runtime for 15.625 µs, s | 218.80 | 216.42 | 219.49 |

Each generalized coordinate was matched by its DOF index. The comparison tolerance was 1% of its original absolute value with a floor of 0.001 in that coordinate’s force or torque units. Both repaired terminal routes cleared the outer skin, dermis–fat interface and deep fat boundary in the matched release pose. A separate kinematic rotation of up to 15 degrees around each of three shoulder axes changed route length by up to 0.79 mm and a rotational moment-arm vector by up to 5.96 mm relative to the original route. This sweep measures geometry only.

The complete variable-pleural native trial retained zero right teres major self-intersections and zero crossings against the three material boundaries at the terminal capture. Five neighboring bones or muscles still intersect its surface in 590 triangle pairs. The registered right scapular endpoint is 50.54 mm from the modeled muscle belly. Continuity of the tendon and belly across that distance remains unverified. The bed carried 11.012 N of 952.864 N. The volume had 37 low-shape cells, 563 principal-stretch failures, four volume failures and 908 cortical samples inside skin or fat. This is an anatomical and force-routing increment in a dynamically released body. Supported equilibrium, sustained throughput and the 16-second coupled motion test remain open.

On SSH Numi, replay with `bash scripts/run_resting_human_on_numi.sh --dense --posterior-support registered-shoulders --steps 2`. The same choice with `--check` verifies all 499 pinned inputs. To reproduce the final NHMYO and NHTENDON from the independently recovered cortical trial, run `rebuild_wrapped_teres.py PARENT_NATIVE_RUN scapular-ports-cortical NEW_OUTPUT_DIRECTORY`. The three recovery manifests in this directory bind the source anatomy, comparison and native motion to independently downloaded GitHub archives.

## Where the posterior preload sends its force

I recomputed the change in elastic traction on every node touched by the 40 mm full-weight preload, including the nodes held fixed in its reduced calculation. The projected bed reaction was 952.86 N. The registered gluteal connection gained 20.14 N on its three skeletal donors. The remaining fixed material nodes received 919.38 N of net incremental traction. The change of internal elastic force summed over all affected nodes closed to within 10⁻¹² N. The two numbers use different force observations and need the direct tissue weight and other reactions to balance.

Of the 1,344 fixed cutoff nodes carrying incremental traction, zero occurred among the generic rigid-contact node IDs at the matched native terminal step. The largest connected group received 822.72 N. Its positive-traction centroid lay 29.33 mm from the registered L3 bone. Another group received 97.35 N and its centroid lay 53.79 mm from the nearest bank owned by the right tibia. This locates the unverified routes into the larger body. Neither distance establishes an attachment.

Allowing sideways material deformation at the existing normal preload reduced its tangential force residual from 897.43 N to 1.45 × 10⁻⁷ N. New stretch failures rose from 312 to 474. The normal deformation already forced 164 new failures even with unrestricted tangential movement. A second solve released all three material coordinates and three gross pose coordinates inside the same truncated domain. It ended with a 419.02 N normal imbalance and 1,096 new stretch failures. Both proposals were withheld from native initialization.

The next equilibrium solve must carry the reaction across the lumbar and lower-leg boundaries through their surrounding material and physical interfaces. A net bed force in the reduced region cannot certify that transfer. The owner tool `check_preload_boundary_traction.py NATIVE_RUN BED_DIRECTORY PRELOAD_DIRECTORY OUTPUT` recomputes the traction. Its optional `--max-held-force-change-n` checks the peak change on a fixed material node. At the 3 µN tolerance used by the regional equilibrium solver, the unloaded displacement control passed and the full-weight proposal failed.
