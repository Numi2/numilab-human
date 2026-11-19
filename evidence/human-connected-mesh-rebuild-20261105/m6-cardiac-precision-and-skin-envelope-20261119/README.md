# Resting Human precision and support diagnostics on Apple M6

The preceding qualified comparison is commit
`944ecd3dea35b94c1f98598bea988097cdabc630`. These are bounded follow-on
experiments. Qualification remains open. The default scene, physical
timestep, pressure coupling, volume tolerance and rollback checks are
unchanged. Physical native trials used the existing articulated and local
skin/fat bed-contact owner. The cardiac authoring probes use the existing
source-field coefficients.

## Native envelope trial

A three-pose exact authoring audit and a separate source audit approved
2,137 source positions for an experimental run. The native scene completed
all 8,000 steps with prescribed exterior pressure and no root assistance.
The 16-second run achieved 0.10752× during motion and 0.09839× from process
launch to exit. It retained three complete breaths and 19 filling/ejection
cycles. The 878-bank terminal audit reported 14,353 crossings, with 2,049
introduced and 2,302 removed relative to the matched 14,606-pair control.
The 55 skin self-pairs were unchanged. The native regression gate rejected
the candidate. Identical compiled bed sections and a passing one-step
check were insufficient to preserve motions and actual support impulses.

A 704-step follow-up held source region IDs fixed and reproduced the
motion divergence. A paired live selector observation isolated the
earliest differing selection at accepted step 385, region 22. The trial
selected source vertex 48742, moved by 0.928 mm in the source, in place of
35615. The point remained above the bed at that step. The first differing
impulse in the ordinary 32-step observer run fell in the interval ending
at physical step 610. Both findings are compatible. The support selector
can change before the corresponding witness becomes load-bearing.

The separate local self-separation trial reduced terminal frozen-pose
skin pairs from 55 to 31 with no new exact pair at the three frozen poses
or source. This is authoring evidence. Its native result requires separate
verification because the actual body trajectory depends on the edited
skin positions. Fully pinned pairs in the retained footprint require
requalification of the supported region when the skin geometry changes.

## Cardiac field and arithmetic

All 434 flagged triangle pairs in the diagnostic 20-pose, prior tight
topology trial intersected with the native rounded positions. The same
pairs were clear with exact arithmetic over the stored Float32
coefficients. One final Float32 round and a fixed local origin both left
witnesses. A two-float Metal evaluation was checked on the original
topology, at 486,556 vertices across 20 probes. The geometry proof counted
zero unresolved pairs at those exact pressure-coordinate points and at
their twenty narrow coordinate intervals after correlation was retained
in the projections. The geometry analysis includes all seven welded
anatomical surfaces and quotients only declared reciprocal interface
facets.

Three selected correlated pressure intervals were fully bounded on the
same source, including the exact native rejected right-atrial coordinate
at 5.560 seconds. The two later intervals bracket the large calibration
excursion near 11.616 seconds. Refinement uses an unchanged cross-face
criterion and a bounded per-pair subdivision. Six synthetic cases check
that real crossings, including one strictly inside an interval, still
fail. The two-float GPU probe and the interval proofs have not been
admitted as the runtime geometry path. The exterior-pressure qualification
at 16 seconds remains outstanding. The last qualifying prefix of the
retained variable-pressure native run is still 5.558 seconds.

The launch receipt, device, asset hashes, terminal trace and comparison
are included. Reports ending in `.json.gz` and `.csv.gz` can be read with
`gzip -dc`. Full source geometry and raw capture files remain in the
documented SSH Numi workspaces. No mesh or solver thresholds were waived.
