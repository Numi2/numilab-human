# Finite-strain support under the resting adult

The existing 38 bed sections cover 0.19568 m² of projected skin. Their linear contact model used moduli of 200 kPa for skin and 55 kPa for fat. The full-volume Human material programs give confined reference tangents of 684.84 and 6.437 kPa. A finite difference of the direct tetrahedral force reproduced each tangent within 9e-12 relative error. Installing these tangents alone put 306 skin vertices more than 1 mm through the bed at 0.062 s. The maximum observed penetration was 15.16 mm.

The new GPU law evaluates the nonlinear compression of the registered skin and fat. The two layers in each column carry equal traction. The 7,282 columns transfer their forces through the existing 38 contact rows and articulated point Jacobians. Independent direct evaluations gave maximum relative interpolation errors of 0.0054% for force and 0.069% for its tangent. The GPU uses the local tangent for the discrete velocity solve.

| Measurement on SSH Numi, Apple M6 | Finite-strain contact |
| --- | ---: |
| Accepted duration | 16.000 s, 8,000 steps |
| Terminal-second sampled support against 952.86 N weight | 953.09 N |
| Maximum sampled vertex count more than 1 mm through bed | 0 |
| Breaths and complete filling/ejection cycles | 3 and 19 |
| Physical skin/anatomy pairs at 7.998 s and 16 s | 11,494 and 11,789 |
| Skin self-intersections at those samples | 32 and 49 |
| Active, integrated body and cold factors | 0.09678×, 0.09394× and 0.08923× |

The dense material admission requires a minimum principal stretch of 2/3. In a laterally confined fat column at the reference state this permits 3.703 kPa normal pressure. Under that model the present sections can sustain at most 520.0 N in static equilibrium within the envelope. Giving every registered column independent compression raises this upper bound to 724.7 N. Both are below body weight. Twelve sections exceeded their static capacities in the final second. Across the captured trajectory, 3,488 columns went below the dense stretch floor. The most compressed fat reached a stretch of 0.067. These strains are inferred from the reduced constitutive model.

At that pressure body weight requires at least 0.25729 m² of engaged projected area. The initial body's entire lower projection is 0.54671 m². In the release posture, the skin within 20 mm of the fixed plane accounts for 0.06533 m². Sufficient projected area first becomes geometrically available about 52 mm above the plane. Actual engagement requires physical deformation and load transfer. Widening the contact band alone would count elevated skin before it takes load.

Launch the development comparison on Numi with `~/Work/run-numi-human-rest --material-contact`. The previous four-patch control is available with `--forearm-contact`. The strict audit options `--require-capacity` and `--require-dense-stretch` reproduce the two material failures. The recovery receipt records the independently downloaded native Metal library, executable, generated material data and accepted geometry.
