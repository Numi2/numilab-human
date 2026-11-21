# Regional triangular factor on Apple M6

The current GPU solves two regional factors with 6,261 free coordinates. Each side passes through more than 3,300 sparse dependency levels. The comparison used the captured 286 ms load residual, a seeded random load and a localized force couple. It used the original factors, permutations, scaling and floating-component projection. Each GPU comparison rotated order across five pairs after warmup.

Precomputing the responses of 32-row triangular diagonal blocks reduced median GPU time from 2.961 ms to 1.331 ms on the captured residual. The two GPU outputs differed by less than 0.00085% in regional displacement L2 norm. The independent double-precision triangular solves and equation residuals are in independent-factor-audit.json. Retaining this test case makes the physical sensitivity reproducible.

## Native decision

The baseline and blocked solve both completed 94 steps with intact cortical geometry at the same physical time of 188 ms. Their matched launch-to-exit wall times were 59.008 and 55.110 s. All reported gas, chamber pressure, chamber volume and ventilation values checked at the common original segment captures matched exactly. The longer native baseline admitted 143 roots. The blocked continuation rejected root 95. Its endpoint geometry had zero transverse pairs, but the loaded-contact gap on the left was 0.5933 μm against a 0.5 μm release limit.

At root 94 the difference between the actual left sacral force vectors was 4.2333 N relative to a control resultant of 60.5556 N. The tissue nodal contact force field changed by 7.643% in L2 norm. Across all matched roots the peak absolute difference in the left sum of pressure magnitudes was 11.840 N, 14.45% of its control peak. This exceeds the 1% force criterion. The experimental blocked solver is archived and the selected native scene uses the original factors.

A second implementation retained the exact per-row floating-point order and used a single SIMD group for each triangular solve. All three output buffers matched byte for byte. The accepted-residual median went from 2.728 to 3.743 ms. The existing launch configuration was faster on this problem.

## Contact conditioning

At the accepted 286 ms endpoint the left regional solver carried 253 positive pressure rows. Only 201 had distinct exact columns after accounting for all tissue nodal forces and each bone's force and moment in the registered pelvic frame. Independent SVD of the row-scaled map measured rank 136 at a relative cutoff of 1e−6 and 149 at 1e−10. The opposite side had 89 distinct, independent rows at both cutoffs. The rank test applies to the observed active set. It does not establish which unilateral inequalities will remain active at another pose. The physical wrench comparison above establishes that the difference reached the skeleton.

Apply experimental.diff.zst to a separate copy of the posterior registration constitutive directory by decompressing and running patch -p1 there. Copy the files from experiment to the registration tools directory alongside material_context.py and the piecewise-linear material audit. Run commands.sh there, passing the motion registry assets, the retained 286 ms capture directory, its partition, a new output directory, and 192 roots. The fixture files contain the measured loads and response vectors.
