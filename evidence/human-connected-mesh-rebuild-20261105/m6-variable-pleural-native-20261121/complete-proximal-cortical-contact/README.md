# Full proximal phalanx surfaces in the articulated contact solve

On SSH Numi, both right and left ring-to-little proximal phalanx pairs remained clear at all 416 measured poses over 0.830 s. Native pre-dynamics frames were captured at each 2 ms step. The emitted terminal mesh provided the independent final pose. The comparison checks the original Float32 cortical triangles of both pairs. Terminal reconstruction differed from the captured vertices by at most 0.068 µm. All 18 paired combinations of ring/little proximal, middle and distal phalanges passed at the terminal native capture. The other 16 combinations were also checked every 32 steps. Neither test establishes clearance between physical steps.

The four fixed-feature rows from the previous trial missed seven triangle pairs at step 352. Its per-step repeat first crossed at step 329 and reached ten pairs. The new calculation evaluates the complete source triangle surfaces for both proximal phalanges of each hand. Each side partitions the first bone's faces into two disjoint sets and matches each set against all faces of its neighbor. The GPU finds real closest witnesses at separated poses and evaluates finite triangle separating axes for intersection. The existing articulated solver applies the normal reaction through the difference of the two material-point Jacobians. At step 352 the right-hand normal impulse measured 0.0223163 N·s. The external observer kept its original 32 bed sections. The GPU triangle kernel also passed 195 independent convex-hull distance cases, including 36 penetrations. Maximum signed distance error was 2.05 nm and maximum witness-line discrepancy was 2.09 nm.

| Result at 0.830 s | Original | Four fixed features | Full proximal surfaces |
| --- | ---: | ---: | ---: |
| Cortical intersections across 18 ring/little bank comparisons | 18 | 0 | 0 |
| Transverse skin self-intersections | 302 | 239 | 274 |
| Transverse skin/nonocular anatomy intersections | 10,127 | 9,977 | 10,014 |

The new native command took 230.953 s including initialization, or 0.00359×. The skin still has no qualified lateral finger-pad material response in this configuration. A clear cortex does not certify the external skin, continuous settling or the complete human.

The retained four-feature 16 s attempt accepted 421 steps (0.842 s) before the common transaction rejected the next left pelvic equilibrium. Its authoritative clocks agreed. The left free-node residual at that attempt was 0.0139975 N against 0.0101567 N allowed, with release gap 0.578 µm against 0.500 µm. Later scratch rows from the same scheduled command buffer are excluded from accepted results. The full proximal surface test retains physically loaded bed sections, six cortical/gluteal checks per accepted endpoint and variable pleural pressure. Full volumetric skin/fat, passive fascia and the loaded aortic wall remain disabled in these runs.

The source-registration directory contains all four face ranges and the complete original source-face ancestry. Build the Human owner with the tiled-forward Human base using the retained launch recipes. Prepare the strict source registration, compile its triangular surface payload, and pass that asset directory as the fourth argument of the trial runner. The adjacent reciprocal-digit-contact evidence retains the superseded fixed-feature trials.
