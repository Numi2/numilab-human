# Matched surface and volume refinement of the connected scalp footprint

Apple M6 on SSH Numi, macOS 27.0 build 26A425. This offline comparison
starts from the verified 09c2fe4 geometry with 16,408 exact whole-shell
triangle pairs by original source-face and bone-triangle identity. The
dominant deep skull component contains 121 parent faces and 377 crossed
child faces. Its measured two-ring neighborhood has 590 eligible existing
columns. The experiment uses the same underlying source skin, native
registered bone bank, source volumes, accepted mass reference, and
density and thickness bounds in both branches.

The refined branch adds matched barycentric nodes on the outer skin,
skin/fat interface, and deep fat boundary. It replaces each corresponding
triangle and splits every incident tetrahedron on both sides of the
interface. It preserves typed ancestry and verifies the signed cavity
boundary in the saved rest and emitted Float32 frames. Ten face columns
fail the existing determinant or shape floor, and 153 would place a new
vertex within a closed skull bank. The 214 groups of three-layer splits
that pass add 642 nodes and 1,712 tetrahedra. Exact unchanged-position
checks preserve all 16,408 original crossing identities. Raw child-face
triangle pairs rise from 16,959 to 17,415 before fitting as expected
from the new tessellation. The fit uses the same trust radii,
determinant/shape admission, two-sided thickness and typed packing
constraints on both topologies.

| Candidate | Outer skull | Interface | Deep | Whole shell | Skull containment | Tets |
|---|---:|---:|---:|---:|---:|---:|
| Starting checkpoint | 480 | 1,244 | 2,504 | 16,408 | 91 | 33,000 |
| Existing topology control | 484 | 1,213 | 2,476 | 16,353 | 89 | 33,000 |
| Paired face and cavity refinement | 480 | 1,177 | 2,459 | 16,296 | 85 | 34,712 |

Refinement clears 57 more crossings and four fewer contained pairs over the
matched control at a cost of 5.2% more tetrahedra. Relative to the starting
checkpoint, it removes 112 whole-shell and 45 deep pairs. Its raw count
after fitting is 17,263 against the subdivided seed's 17,415. Reported
containment covers each independently closed skull object and all shell
boundary vertices. No new contained pairs, other-bank crossings, or local
self-intersections survive the audit. Outer skull crossings are unchanged
at 480. All 34,712 cells remain positive in the independently verified
rest and emitted frames. The three bed support vertex coordinates remain
identical to source. The allocator conserves 13.939679838734492 kg with
typed donor bounds and a maximum independently reconstructed centroid
route of 69.492 mm under the original 75 mm limit.

The paired fit accepted five steps at fractions from 1/16 to 1/8.
Exact determinant floors were the most frequent geometry blocker. Repeated
trial failures record 492 below-floor cell visits per frame. The retained
worst rows include 81 visits to newly split cells and 91 to cells with
unchanged topology. They span 50 distinct child rows. The matched status
shows broad face insertion alone leaves the determinant bottleneck.
Both branches use the original density ranges and force model.

The first refined invocation exited before saving a candidate. Mass
allocation incorrectly attempted identity routes when the new trial made
no deformation, ignoring the resumed checkpoint's redistributed mass.
The final tool explicitly includes already-routed and already-moved
source owners in the conservative allocation. Both tested solvers on
unrefined topology reached identical crossing and containment counts.
The initial unrefined run used default HiGHS and took 80.65 s. The final
same-solver runs used HiGHS IPM and took 128.68 s unrefined and 148.76 s
refined, including exact offline auditing. These are individual
preparation measurements. Neither model was timed in native motion.

The refined branch is an offline candidate. Native startup, full tissue
contact, loaded shoulder behavior, and bed equilibrium have not been
qualified for it. The 16-second variable-pleural-pressure test, zero
crossings through motion, and end-to-end 0.1× and 1× remain open.

## Reproduce

Restore the parent experiment `cavity-batch-037` from its checkpoint on
SSH Numi. Create adjacent `surface-patch-038` and copy `rebuild.py`,
`fit_layers.py`, `column_motion.py`, `skull_contact.py`,
`solid_skull.py`, `regional_mass.py` and `verify_rebuild.py` from the
parent. Add the files in this publication and apply `lp-method.patch`.
Use the existing preparation venv.

```sh
python3 test_paired_surface.py
python3 run_patch.py refined --refine
python3 run_patch.py control
python3 verify_patch.py refined control
python3 audit_patch_self.py refined
```

The private asset in the recovery receipt includes the matched meshes,
lossless numerical receipts and the failed attempt witness. The public
experiment contains the procedural code, results and hashes.
