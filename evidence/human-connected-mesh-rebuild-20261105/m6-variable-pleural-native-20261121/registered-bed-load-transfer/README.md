# Registered skin contact and placement on the supported body

The native Metal contact selector now takes its candidate positions and Jacobians from the registered outer skin. The measured scene uses 76,153 vertices and 32 anatomical contact regions. Each section retains its original area, stiffness, thickness and damping. Articulated dynamics, local gluteal tissue, circulation, respiration and Brain advance in their existing owners. The separate full-volume skin/fat solver remains unqualified for this pose.

All placement trials started from the same 188 ms articulated posture and pelvic material construction, with a fresh physiology clock and initial joint velocities. The free root was lowered 1.47 mm and rotated about the measured center of mass by 0.014° and 0.089°. The native simulation then reselected its surface points. No external root force or wrist/joint lock was used.

| First accepted native step | Bed normal | Normal mapped back to physical skin, conserving section force and moment |
| --- | ---: | ---: |
| Previous contact positions | 919.42 N | 407.46 N |
| Registered skin at the same placement | 543.54 N | 543.54 N |
| Registered skin and balanced placement | 952.62 N | 952.62 N |

Body weight was 952.86 N. The independent map clips triangle quadrature to the plane, resolves occlusion and checks conservative P1 transfer. For the seated candidate, the uncompressed predicted footprint was 59.60 cm² and the minimum possible peak pressure for the aggregate wrench was 318 kPa. This is an inverse construction on one undeformed footprint. The contact model does not measure dense tissue pressure. Registered-source point reconstruction error at the original posture was below 0.36 µm for all 32 regions.

The paired 32-step native run accepted 64 ms. Its body interval took 19.10 s (0.00335×). Launch to exit took 27.55 s (0.00232×). The associated 16-second attempt stopped with 46 accepted 2 ms steps and matched accepted mechanics, respiration and Brain clocks at 92 ms. The pelvic certificate recorded zero transverse pairs in the accepted prefix. Its rejected candidate had geometry status bit 2, indicating motion-registry overflow on this shader. The flag column is historically named 'nonfinite'; bit 1 records nonfinite geometry. The accepted-prefix claim covers the tested pelvic surfaces only. The whole-body crossing, 16-second, throughput and settled-state gates remain open.

The dense reference geometry was tested separately at the unseated imported posture. All 10,755 inverted cells in the rigid per-node guess crossed a donor boundary. A continuous one-ring gradient fit reduced this to 3,221. A separate 18 mm smooth-field trial at the balanced posture had 931 inversions and a maximum 27.17 mm mismatch to the per-donor target. Neither field was installed in the simulation. Mesh connectivity, cell mass and material reference positions are preserved in these construction experiments.

## Reproduction

Use the Apple M6 and the source and toolchain paths in the retained configs. Restore the original body and supported native build from the parent evidence. Build the selector using install_registered_bed_kinematics.py with the tiled-forward native build, native app and a fresh destination. The build helper checks the parent binary and writes exact compiler commands and output digests. The fit script takes the matched one-step capture, donor capture, base config and output directory. The pinned run_cortical_trial.sh entrypoint consumes the resulting config, source asset directory, binary, output directory, prepared Python interpreter, step budget and enable flag. The retained launch hashes identify every input. The registered load quadrature, rigid-motion and clipping tests are alongside the builder. Gzip archives were decompressed and hashed against their original bytes.
