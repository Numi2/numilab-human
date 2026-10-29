# Conforming connected skin/fat refinement

This increment provides offline conforming edge bisection inside the existing skin/fat preparation path. It appends source midpoints, splits complete incident tetrahedron stars across material interfaces, preserves original node IDs, and records source-node barycentric CSR plus exact original-cell measure fractions. Runtime tissue coordinates stay independent.

## Actual SSH Numi control

Exact component ancestry maps the largest retained body 23/bone 29 overlap to 28 original I triangles and 31 original D triangles. The I seed reproduces the prior 548-subface identity map exactly. Their combined graph has 86 local edges, refined on all three O/I/D layers.

Source 002 grows from 7,233 nodes/28,632 cells to 7,491 nodes/29,642 cells in 0.144 seconds of offline preparation. Original nodes remain unchanged. Source-reference and emitted Float32 cells are all positive. Original-parent mass closure is exactly 0 kg; the maximum relative source-cell volume-sum error is 2.29e-15. These are construction timings, not simulation throughput.

The topology checks find no face incidence above 2, paired-face orientation error, outer-volume boundary edge error, or shared-I role error. Both 9,988-face material boundaries have zero exact self-intersections using canonical Float32 coordinates, exact integer predicates and no seam exemptions.

Portable tests cover unchanged original nodes, exact source measure closure, barycentric reconstruction, matching skin/fat interface subdivision, nested refinement, duplicate marked edges and invalid/inverted input rejection. Source, seed, owner and output hashes are retained in the receipts.

## Current admission boundary

The control has no anatomical target projection and has not entered the native runtime. The existing source geometry still needs connected anatomical fitting, full nodal/label/attachment metadata, native mass import, bone/contact checks and accepted motion. No runtime, GPU step, performance or physiological qualification is claimed here. Current fine anatomy and the working back repair remain outside this change.

Two original deep attachment sites are inside owned bone banks: volume node 6155 is 1.537 mm inside bone 33; node 5626 is 1.354 mm inside bone 132. The other 12 sites have no owned-bank containment and lie within 0.121 micrometres of the full fine D boundary. Their binding identities have not changed. The retained donor-only closest-surface lookup is a failed diagnostic control: derived faces may carry body=-1 with valid full ancestry, so filtering only by that scalar can choose a distant surface. Its points must not be used as replacement attachment targets. The addendum makes this explicit.

The new native reference-pullback owner and actual source-only import controls are separately published in Numi Lab as [456b1ee9797737f1b5dd4dc2a944cfaa56d9b2e1](https://github.com/Numi2/numi-lab/commit/456b1ee9797737f1b5dd4dc2a944cfaa56d9b2e1). They preserve immutable 330 source mass; the connected projected mesh still needs that admission chain.
