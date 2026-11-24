# Bilateral thumb-abductor anatomy and physical force transfer

The published supported-shoulder scene is the control. Two independent native experiments changed different parts of that scene on SSH Numi. They use the same regional skin/fat bed support, variable pleural pressure, native cardiac solver and articulated body. Whole-body passive fascia is disabled. The dense skin/fat owner still rejects its third step in the inherited control.

| 16 s native result | Original | Muscle-surface binding only | Physical tendon routes only |
| --- | ---: | ---: | ---: |
| Accepted steps | 8,000 | 8,000 | 8,000 |
| Whole-body skin intersection pairs at 16 s | 11,531 | 11,374 | 11,553 |
| Skin self-intersection pairs at 16 s | 24 | 24 | 24 |
| Cold process real-time factor | 0.09284× | 0.09272× | 0.09268× |

The original two-owner muscle binding gives about 87% thumb motion to material near a guide fixed to the radius. Interpolation along the original actuator route removes 96 right and 109 left muscle self-intersection pairs at 16 s. The bilateral APL-to-skin pair count drops from 295 to 138. Right APL then overlaps the extensor carpi radialis brevis by 5.33 cm³ and extensor digitorum by 4.46 cm³. The muscle binding trial fails neighboring muscle clearance and remains experimental. The archive retains the original folded surfaces, Boolean normalization details, source volumes and four muscle candidates.

Separately, 131.74 mm of the original right APL and 134.99 mm of the left APL force path lie outside the skin in the final control capture, using samples spaced at most 1 mm apart. The physical experiment changed the sixteen existing private MyoSim sites, moved four resultants to their own registered cortical triangles and recomputed the tendon force maps on exactly the same sixteen cortical load sites. The largest resultant displacement was 3.23 mm. All four serialized force maps conserve an arbitrary terminal force to within 1.6e−8 N per newton and moment to within 1.9e−10 N·m per newton. The GPU uses the existing constitutive laws and load owners.

The physical experiment removes external-skin centerline crossings in both routes at release, 7.998 s and 16 s. The complete regional run has 3 breaths, 19 sampled LV pressure peaks above 70 mmHg, 1,352.35 ml cumulative aortic ejection, zero root assistance and a last-two-seconds sampled mean bed reaction of 952.86 N. It takes 172.64 s cold for 16.00 s simulated, including native presentation. The measured body loop takes 164.20 s.

All 250 retained mechanical poses were independently checked against cortical surfaces in their own rigid coordinates. The first physical trial still enters bone at 41 right and 85 left samples. The maximum in-bone intervals are 9.81 mm and 2.23 mm. Close agreement with fixed-time captures therefore cannot certify a moving tendon. These are centerline tests. Finite tendon thickness, its retinacular relationship, gliding contact and the surrounding muscle/fat interfaces require separate qualification.

The exact compiled programs, inputs, diagnostics, failed fits, accepted traces and geometry are in the independently restored GitHub release assets listed in the recovery receipts. The unchanged baseline can be restored through the parent recovery receipt. The physical and surface experiments remain separate from the default scene.

## Native route verification through settling

A second physical trial incorporates the cortical failures found during early settling. The same two MyoSim actuators and four distributed tendon footprints apply the new loads. In the new 8,000-step run both tendon centerlines pass the cortical check at **all 250 retained native mechanical samples**. Each segment must clear every potentially intersecting cortical triangle. An endpoint within 1 micrometre of its own cortical triangle is classified as attached. The largest such interval was below 0.00009 mm.

Neither centerline exits the skin at the seven checked poses (release and steps 223, 447, 575, 1216, 3999 and 8000). The narrowest sampled skin margin is 0.68 mm. The four early poses come from a native prefix with identical values in all 5,966 retained body rows and all 38 coupled physiology rows shared with the full trial. These measurements leave tendon thickness and clearance between retained poses unqualified.

The complete regional run took **172.224 s cold for 16.000 s simulated (0.09290×)**, including 252 presentation calls. The body loop was 0.09770×. The bed carried a sampled mean 952.88 N over the final two seconds. The largest final body translation difference against the matched original was 0.56 mm. The eleven compared cardiopulmonary channels agree at the saved times.

The full-body skin audit reports **11,508 pairs at 7.998 s and 11,552 at 16 s**. The matched control reports 11,514 and 11,531, respectively. The final skin self-intersection count remains 24. APL clearance is a local physical improvement. The default anatomy release remains open.

The source-coordinate inspection also separates original anatomy from deformation error. Right APL/extensor carpi radialis brevis overlap is 3.35 mm³ before posing and 5,326 mm³ in the surface-binding trial. Right APL/extensor digitorum overlap is 205 mm³ before posing and 4,464 mm³ in that trial. Both native registrations use the same material-to-radius transform. Independent muscle weights amplify the source overlaps. Reconstruct the adjacent extensor compartment with one deformation and preserve each muscle's individual attachment and force owner.

The regional and dense models share the same cortical triangulations and source-frame guide coordinates. Transferring the successful regional path into the captured dense pose still leaves 3.41 mm of the right path and 29.86 mm of the left path outside the deep fat boundary, together with a 0.82 mm interval in a right carpal bone. This is a dense geometry preflight. The complete dense contact scene has not qualified through motion.

Run the physical trial from its installed Numi directory with the following command.

~~~sh
ssh numi 'cd ~/Work/numi-human-supported-shoulder-085/force-route-086 && ./run-supported-scene --launch "$PWD/connected-forearms-087/refitted-source/launch.json"'
~~~

The moving recovery receipt identifies the peer-downloaded release archive and independently verified members. Identical files shared across runs are resolved through its inherited-file index.
