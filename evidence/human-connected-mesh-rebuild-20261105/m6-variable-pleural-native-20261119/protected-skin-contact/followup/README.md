# Exact support-edge dependencies and shared contact trials

The 16-second recorded reference retains 14,207 skin-to-tissue pairs and 45 skin self-intersection pairs at the terminal capture. Re-running the same geometry with the separate shared FEM owner reached 4 ms with normal gravity and 6 ms with the explicit startup gravity ramp. The ramp test failed the reference nonlinear residual on the first full-gravity step. Both used variable pleural pressure and the registered ligaments. Neither qualifies tissue contact through sustained motion.

## Shape dependencies

The 17-position face plan still leaves 24 witnesses forced by fixed edges. The exact endpoint audit uses Float32 lattice intersection geometry and the native one-based source-index conversion. It finds that 26 thickness or guard positions would have to be released to give the relevant crossing endpoints freedom. One intersection between face 14040 and tissue bank 51005/121 at triangle 1476 lies on the edge between source rows 6889 and 7240. Both rows are actual registered bed support anchors. Releasing thickness samples cannot move that edge. Screening the larger position-only proposal on the two retained poses produced zero admissible patches.

## Native solver measurement

The Newton-tail indirect graph kept the same first two accepted steps and five byte-identical trace files. It used four graph commands including the two tangent-cache commands. Their combined GPU time fell from 36.071 to 30.600 s, a 15.2% reduction. The third step rejected the same cell and diagnostics. This short trial measures the cost of the accepted prefix. The shared contact path still costs about 15 s per 2 ms step, or about 0.00013× during that prefix.

The constitutive-aware backtracking trial at the parent checkpoint enabled the existing non-residual backtracking path. The evidence establishes identical rejection with that setting. It did not test or establish that residual Armijo ran on this shared topology.

The two-face diagonal replacement preserved both support anchors and passed the source orientation check. At 16 seconds it increased the target intersection-curve length by 7.8 µm and produced new triangle-pair IDs in two bone banks. It was excluded. Joint surface and support registration is still needed for that edge.
