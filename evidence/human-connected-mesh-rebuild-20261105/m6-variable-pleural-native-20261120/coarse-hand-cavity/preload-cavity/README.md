# Boundary-preserving posterior fat remesh

A fresh native-derived control remained geometrically limited to 2.5%
of 952.864 N total weight in the restricted offline bed preload test.
At the failed 5% trial, adipose tetrahedron 21848 reached the upper
determinant guard. Four direct adjacent flips failed the cavity orientation
check. The first feasible same-bank edge-star remesh replaces four cells
and leaves every vertex and oriented cavity boundary unchanged. At the
saved terminal deformation, its four new cells stay between J=0.50986
and 0.71814 instead of reaching the expansion guard.

The replacement uses reference-volume intersection to transfer each
previous cell's source-carrier masses. Row and column volume close in
the same geometry. The largest error per retained source mass is
1.36e-20 kg, with 13.939679839 kg total and the same tissue, anatomical
region, and articulated donor bank. A zero-thickness shared-face case
and a known oriented cavity with a conservative flip passed on Apple M6.

The matched preload test uses the original complete material energy,
bed law, force gates, and exact determinant segment search. It converges
at 5% gravity, with 47.6432 N across 145 active sites. The control
converges at 2.5% and rejects 5%. At the 5% solution, all 28,632 cells
are positive and above the normalized shape floor, and there are zero
shell self- or cross-layer pairs. The Float32 minimum volume ratio is
0.40920. The 7.5% attempt ends at cell 22050 near the compression guard
without passing the force residual. The 5% deformation produces 26
new bone triangle-pair witnesses and removes 12. That fails the starting-pose
clearance gate. The native run starts from the original pose.

The native experiment applies the changed connectivity and mapped mass
at the original startup geometry.

## Native release replay on Apple M6

The existing Metal runtime accepted 128 coupled roots at 15.625 µs.
Pleural pressure varied from −490.332581 to −490.332489 Pa. The
registered reference shape and analytic bed remain fixed. The compiler
preserved source-face ancestry through the original parents, using
overlap fractions for volume and exact per-facet IDs for the layer
surfaces. The two rejected startup attempts identified the input
rebase and the missing stable-face table. The final build uses the
recovered earlier native control to verify geometry and dynamics.

| Native measurement | Previous registered control | Four-cell remesh |
| --- | ---: | ---: |
| Accepted time | 2 ms | 2 ms |
| Body segment wall time | 98.765 s | 97.970 s |
| Body real-time factor | 0.00002025× | 0.00002041× |
| Launch to exit | 114.196 s | 113.403 s |
| Last bed reaction | 0.44345 N | 0.43787 N |
| Initial bone intersection pairs | 17,850 | 17,850 |
| Terminal bone intersection pairs | 17,897 | 17,897 |

Every audited bone pair identity was unchanged in both comparisons.
The initial Float32 geometry matched the recovered native control
exactly. The seven z-coordinate entries rounded differently from the
offline authoring input by at most 3.73 nm. The four changed emitted
cell rows, all typed donor/region/material banks, and full mass mapping
passed independent checks. All 34 ligament rows passed the terminal
audit, and five captured poses have zero unresolved bilateral shoulder
path crossings. All 28,632 skin/fat cells were positive and above the
shape floor at both audited endpoints. There were zero shell self- or
cross-layer pairs at those endpoints.

This separate headless build uses real bed contact and zero root assistance.
The bed took less than 1 N against the roughly 953 N weight. The single
timing pair measures this short release. Settled equilibrium, continuous
16-second clearance and full-scene rates of 0.1× and 1× remain open.

## Reproduction and provenance

The commands in the pinned build and launch receipts use SSH Numi's
retained authoring checkout, asset inventory and existing runtime.
Build with `build_native.py`, the previous installed build, the previously
rebased source ledger, the overlap remap, and the retained exact
`NumiHumanElbowExplicitPairedTopology008.hpp` table. Verify against the
independently recovered previous native checkpoint. The probe keeps
the native certification gate and anatomical ownership. The study
includes the exact validated build pins, startup identities, stopped
preload states, native pressure trace and audit output.

BodyParts3D Copyright © 2008 The Database Center for Life Science,
licensed under CC Attribution-Share Alike 2.1 Japan. The skin/fat
reference is assembled from registered sources. See the earlier
posterior-bed evidence for the source licenses and inference boundaries.

The native viewer pack is byte-identical to the control. The separate verified
archive contains the changed native executable, generated registration, emitted
tissue tables, coupled trace, and full numerical witnesses.
