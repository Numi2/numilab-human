# Fixed-volume comparison during supported release

The previous fitted release allowed the limiting fat cell to reach 32.889% of the original starting volume. This preparation holds the previous volume while fitting the complete connected stars of the affected remeshed material nodes. It moved 280 existing nodes, preserved both external material surfaces and all 14 attachment sites, and changed the limiting cell’s initial volume by 0.008426%. All 1,448,799 installed cells passed preparation. The maximum installed nodal versus cell mass-transport difference was 2.287e-21 kg m.

The native M6 run accepted 3 of three steps, corresponding to 46.87500 µs in 250.333 wall seconds. All release comparisons used the same bed, gravity, constitutive laws, recruitment bounds and numerical acceptance tolerances.

At 46.87500 µs the minimum volume ratios were 0.350029962 relative to the previous starting shape and 0.35000047 relative to the fitted shape. The respective counts at or below 0.35 were 0 and 0. The bed reaction was 1752.900 N against 952.864 N of weight. Upward tissue force on the 811.448 N skeleton was 2.172 N. Throughput including launch was 1.8725e-07×. The measured native horizon ran at 4.7254e-07×.

The accepted endpoint had 10,894 exterior tissue intersections, 0 skin self-intersections and 0 layer crossings. Element stretch counts are listed separately in result.json.

The native registration snapshot and actual release root differ by 11.718750 mm. The observed starting poses were emitted directly from the 157-body Metal articulated result used by the renderer. Root discrepancy after correction was 6.24e-11 m. The exact executed-pose opening mesh had 10,898 skin/tissue crossings across 878 banks, including 0 skin/bone pairs.

Of the opening triangle pairs, 10,887 also intersected at the verified endpoint. 11 cleared and 7 appeared. The opening uses the source pack and executed body transforms. The terminal mesh uses the accepted GPU vertex buffer. Bank displacement is independently quantified in bank-motion.json.

The two extensor digitorum and two brachioradialis surfaces crossed the original skin in 706 pairs affecting 522 muscle triangles. In the registered skin these became 2,854 pairs across 942 muscle triangles. Exactly 2,854 of these registered pairs appeared in the installed initial skin. The input surfaces already conflict. The additional conflicts arose during registration, before bed loading. Pair counts between different skin topologies cannot be subtracted to count physical defects. Muscle triangles retain common identities throughout this comparison.

The six-step request accepted 3 steps in 276.979 seconds including native launch. The first 3 accepted skeletal load samples and respiratory trace rows reproduced the short run. Accepted time was 46.8750 µs.

At the next step the native solver rejected object 15, index 5, code 10. The common coordinates and respiratory state rolled back. The native diagnostics were [2.500000238418579, 2.5, 950449, 4].

The two native volume witnesses were both subcutaneous fat tetrahedra with all four corners on the bonded interface in anatomical donor 128. Their centers were 50.06 mm apart. Reference altitudes were 21.45 and 18.50 µm. By step three the second was already 2.4999992 times its material-reference volume, with a largest principal stretch of 2.747. Step four exceeded the unchanged material determinant limit of 2.5 during line search.

Use the coupled compact scene for cardiopulmonary regression. The dense forearms require a joint source and registration repair against the executed skeleton. Posterior tissue requires a material-interface and load-transfer repair covering both opposite volume failures. Global tolerances, cosmetic offsets and a longer release cannot establish either result.

These captures cover a short release under gravity. Supported equilibrium, continuous motion with clear anatomy, breathing and cardiac cycles, the gas-response intervention and the end-to-end throughput targets remain to be demonstrated.
