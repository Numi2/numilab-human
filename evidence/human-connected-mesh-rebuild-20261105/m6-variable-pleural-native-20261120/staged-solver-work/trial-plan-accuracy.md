# Matched Newton staging with unchanged Krylov depth

The first staged trial reduced encoded restart work, and its physiology and motion traces first changed at step seven. Four Newton passes accepted step 16, while step 17 required the full fallback. This comparison restores the original Krylov budget for every attempt. Only the staged Newton limits change, with an 8-pass middle tier between the 2- and 24-pass limits and the same rollback proof and acceptance checks.

Compare the same 18 roots, immutable input pins, 2 ms physical timestep, 256 tissue microsteps, variable pleural pressure and paired accepted traces. Predict a clean native exit, ≥1.30× launch-to-exit speedup over the retained 308.568 s, and byte-identical physiology, bed-load and momentum traces. Record every failure, including exact checkpoint rollback. If traces differ, retain the data and withhold an equivalence claim. The process deadline is 420 s.
