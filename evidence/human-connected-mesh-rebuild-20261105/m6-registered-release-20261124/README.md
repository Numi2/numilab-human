# Registered tissue volume during native line search

The original final Metal check computed the oriented volume from compensated world coordinates. The trial search computed its release check through the deformation gradient. On one fat tetrahedron the older search admitted a trial whose represented volume was 0.34999835 of the registered volume. The required fraction is strictly above 0.35. The updated parallel trial search, reduced-alpha search, active direction and terminal check use the same geometric predicate. The constitutive and swept-segment conditions also apply.

The actual Metal helper passed 26 exact-rational comparisons on Apple M6, including reversed orientation, equality, sub-ULP displacement, legacy and reference coordinates, and two anatomical tetrahedra. The seven-node right-ulna registration changes the bonded internal interface by at most 6.47 μm. It retains the left-arm registration, material reference, mass allocations, boundary coordinates and cortical samples.

| Native run | Accepted steps | Simulated μs | Launch seconds | Second mechanical residual |
| --- | ---: | ---: | ---: | ---: |
| control | 1 | 7.813 | 212.26 | 0.000102514168 |
| bilateral | 1 | 7.813 | 215.44 | 7.72495041e-05 |
| consistent geometry | 2 | 15.625 | 221.56 | 7.72497151e-05 |
| distributed registration | 1 | 7.813 | 213.06 | 0.000101642792 |

The two-step run took 73.01 seconds in the coupled body and 221.56 seconds including preparation and capture. The measured factors are 2.14e-07× and 7.05e-08× respectively. The body and physiological clocks agree. This interval is too short for a heartbeat or breath.

At its accepted endpoint the four material surfaces had 0 intersections. Both materials had zero inverted cells. 6 skin cells and 34 fat cells fell below the shape threshold. 39 of the 40 affected cells were within a factor of two of that threshold in the reference geometry. Pairwise cortical overlaps across 185 bones sum to 5.9764 cm³.

An additional connected registration passed the initial material, surface and local stretch checks but rejected the second mechanical step. Two material-interior sampling adjustments corrected 2 captured quality failures in an offline reconstruction and passed local virtual-work, energy and mass transfer checks. Their native motion requires a fresh run. The bilateral two-step configuration remains the reproducible comparison. The 16-second motion, whole-body clearance and 0.1× requirements remain open.

The release contains the exact launch inputs by digest, executables, shader patch, compact accepted-state records, rejected runs, native logs, reconstruction tools and independent SHA-256 recovery evidence. It uses the pinned parent scene and source-owned assets.
