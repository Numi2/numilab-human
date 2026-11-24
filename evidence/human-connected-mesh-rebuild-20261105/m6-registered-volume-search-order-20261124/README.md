# What stops the dense registered skin and fat

A matched three-step trial on SSH Numi tested whether choosing the cell that limits the admissible Newton step could resolve the dense scene’s third-step rejection. The candidate orders eligible cells inside a sparse tile by their admissible dyadic step, with overshoot as a tie breaker. It uses the same GPU direction projection, force equations, material domain, line search and fresh residual certificate. The change was confined to an experimental Metal library.

| Native measure | Unchanged solver | Candidate |
| --- | ---: | ---: |
| Accepted simulated time | 15.625 µs | 15.625 µs |
| Newton iterations in the rejected third root | 16 | 16 |
| Active FGMRES cycles / columns | 46 / 551 | 46 / 551 |
| Exhausted linear solves | 0 | 0 |
| Final nonvascular residual | 0.03170135 | 0.03170135 |
| Newton tolerance | 0.000010 | 0.000010 |
| Last admissible Newton fraction | 2⁻²³ | 2⁻²³ |
| Cold wall time | 256.86 s | 259.61 s |

The physical acceptance result, limiting cell, and recorded nonlinear progression agree. The three global full-step witnesses in the final iteration were 140978, 616008 and 1443118. At a quarter step only 140978 remained. The cell reached the determinant floor of 0.35 during the rejected Newton iteration. Its undamped proposal would have reached −0.03859. At the last four iterations the accepted line-search fraction was 2⁻²³. Tile 537’s direction projection reported success, but the subsequent geometric test continued to see the blocking cell. The observation establishes a failing nonlinear search direction. It leaves the source of the incompatible correction open.

The local-row material join ties the first witness to registered element 137663, material 15, anatomical region 42 and donor body 46. Its material import volume is 6.9853e−17 m³ and reference mass is 6.2868e−14 kg. In the saved two-step physical control its determinant ratio to reference is 0.979016. None of the three full-step witnesses is one of the 37 previous shape failures. Thus clearing that list alone cannot certify the subsequent Newton step. The native material import accounts for 3.452571 kg of skin and 10.487108 kg of fat. All reference cells passed the import geometry and mass checks. The measured two-step physical control subsequently had four skin and 33 fat shape failures and zero inversions.

Two material-authoring experiments also finished. Conforming longest-edge subdivisions found no passing repair of the recorded defects in three matched material frames. Interior material reparameterization examined 14 available interior nodes. Two intermediate proposals passed shape and force comparisons but failed subsequent mass-tag or density checks. Including those physical constraints gave zero admissible changes. Neither experiment changed the running skin or fat.

The registered bed-contact scene and its launchable 16-second benchmark are documented in the [supported motion evidence](../m6-supported-motion-traction-20261124/README.md). The priority experiment stays out of that solver. The resolution target for the dense path is joint compatibility of the active material volume constraints and the current mechanical residual. Its next test should record the exact cell and nodal direction immediately before and after projection and at the line search, including displacement precision and neighboring constraints. Repeating the global Krylov budget search does not test this discrepancy.

The recovery receipt records the independently downloaded archive, reconstructed control and experimental shader specifications, candidate native Metal library, both full solver logs and the failed material trials.
