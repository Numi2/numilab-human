# Conforming connected skin/fat refinement

This increment provides offline conforming edge bisection inside the existing skin/fat preparation path. It appends source midpoints, splits complete incident tetrahedron stars across material interfaces, preserves original node IDs, and records source-node barycentric CSR plus exact original-cell measure fractions. Runtime tissue coordinates stay independent.

## Actual SSH Numi control

Exact component ancestry maps the largest retained body 23/bone 29 overlap to 28 original I triangles and 31 original D triangles. The I seed reproduces the prior 548-subface identity map exactly. Their combined graph has 86 local edges, refined on all three O/I/D layers.

Source 002 grows from 7233 nodes/28632 cells to 7491 nodes/29642 cells in 0.144 seconds of offline preparation. Original nodes remain unchanged. Source-reference and emitted Float 32 cells are all positive. Original-parent mass closure is exactly 0 kg; the maximum relative source-cell volume-sum error is 2.29 e-15. These are construction timings, not simulation throughput.

The topology checks find no face incidence above 2, paired-face orientation error, outer-volume boundary edge error, or shared-I role error. Both 9988-face material boundaries have zero exact self-intersections using canonical Float 32 coordinates, exact integer predicates and no seam exemptions.

Portable tests cover unchanged original nodes, exact source measure closure, barycentric reconstruction, matching skin/fat interface subdivision, nested refinement, duplicate marked edges and invalid/inverted input rejection. Source, seed, owner and output hashes are retained in the receipts.

## Current admission boundary

The control has no anatomical target projection and has not entered the native runtime. The existing source geometry still needs connected anatomical fitting, full nodal/label/attachment metadata, native mass import, bone/contact checks and accepted motion. No runtime, GPU step, performance or physiological qualification is claimed here. Current fine anatomy and the working back repair remain outside this change.

Two original deep attachment sites are inside owned bone banks: volume node 6155 is 1.537 mm inside bone 33; node 5626 is 1.354 mm inside bone 132. Other 12 sites have no owned-bank containment and lie within 0.121 micrometres of the full fine D boundary. Their binding identities have not changed. The retained donor-only closest-surface lookup is a failed diagnostic control: derived faces may carry body=-1 with valid full ancestry, so filtering only by that scalar can choose a distant surface. Its points must not be used as replacement attachment targets. The addendum makes this explicit.

The new native reference-pullback owner and actual source-only import controls are separately published in Numi Lab as 456 b 1 ee 9797737 f 1 b 5 dd 4 dc 2 a 944 cfaa 56 d 9 b 2 e 1. They preserve immutable 330 source mass; the connected projected mesh still needs that admission chain.
