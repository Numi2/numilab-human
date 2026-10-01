# Open Knee source-equivalent subsystem: reference admission

The shifted-ACL whole-body case is preserved as
`oks003-shifted-acl-reduced-hybrid-eight-microseconds` in
[`config/open-knee-shifted-acl-hybrid-regression.v1.json`](../config/open-knee-shifted-acl-hybrid-regression.v1.json).
Its source-bound receipt and binaries are unchanged. Eight accepted 1 us steps
are a contact regression, not evidence of source-equivalent knee mechanics.

## Implemented authoring boundary

`Source.mechanical_program` now retains the entire pinned FEBio XML program.
The new `open-knee-reference-case` command freezes the four original files,
compiles every XML element/attribute/text value in order, checks mechanical
references, and records the source rigid-joint graph. It requires no MyoSim
registration and makes no coordinate, constitutive, loading, or contact changes.
The program includes nested materials and prestrain, per-element meniscus
fibres, rigid-body definitions and ties, discrete springs, contact enforcement,
load curves, solver controls, and requested outputs.

This is a source-program compiler and dependency audit. Matter lowering is
explicitly unsupported, all physical qualification flags are false, and the
reference solver has **not run**. The command cannot promote its output to a
native mechanics result. The existing NHKNEE1 payload remains the reduced
hybrid and its generated manifest now states that boundary.

```sh
PYTHONPATH=src .venv-mujoco312/bin/python -m numilab_human.cli \
  open-knee-reference-case \
  --open-knee Sources/open-knee-oks003 \
  --matter-root /Users/home/MetalRobo-human-coupled-current \
  --output Build/open-knee-reference-20261001/retained-source-audit
```

The output directory must be new. Expected exit status for the retained bundle
is **2**, with a receipt containing all six unresolved references. A failed
admission still preserves the diagnostic artifacts. The original material
reference coordinates are retained byte-for-byte; no initialization solve or
energy reset occurs.

## New source-integrity findings

The original deck and original retained geometry do not form a complete
reference problem:

| Required construct | Retained bundle finding |
| --- | --- |
| `Geometry_custom.feb` | Missing; the deck names an absolute Windows path to this file |
| `QAT_@_QSO_TiesNodes` | Missing from retained `Geometry.feb` |
| `MPFL` discrete set | Missing |
| `LPFL` discrete set | Missing |
| `MCL_MNS-M_tie` discrete set | Missing |
| `FMC_To_QAT` contact pair | Missing; `QAT_To_FMC` is a different, unused definition |

The deck executes 18 sliding-elastic contact entries; the retained geometry
names 19 surface pairs. `MCL_To_MNS-M` and `QAT_To_FMC` are unused by those
contact entries. These counts must not be conflated.

Every active contact explicitly has `laugon=0`, `two_pass=1`,
`auto_penalty=1`, and `penalty=0.1`. Node relocation is not explicitly set in
this deck; its behavior cannot be inferred without the pinned solver's default.
The authored step is static: 40 increments of 0.05, adaptive bounds
0.001–0.05. Prestrain curves ramp ACL to 1.016, MCL to 1.034, LCL to 1.027
between source times 0 and 1. QAT, PCL and PTL curves remain at 1.0. The
flexion load curve stays zero to time 1 then reaches one at time 2, scaling
`Extension_Flexion` rotation to **-1.57 radians**. Those are static continuation
parameters, not evidence of two seconds of physical dynamics. There is no
explicit quadriceps force-loading section in this passive-flexion deck.

The patellar graph is `FMB(4) → PFFO(20) → PFPO(19) → PTB(1)`.
The neutral spatial motion map has rank six. This is a calculation from source
axes/origins, not validation of flexed body transforms, joint reactions, or a
native knee solve. All original intermediate bodies remain represented.

## Upstream recovery

The [DOI archive](https://doi.org/10.18735/b0zv-n395) lists the missing
`oks003/final model/Geometry_custom.feb` (119,032,115 bytes), original
`FeBio_custom.log` (1,472,687 bytes), and `FeBio_custom.xplt`
(5,345,476,723 bytes). Its browser download requires explicit license acceptance.
The download page displays an MIT agreement; the retained `license.txt` is
CC BY 4.0. Preserve both notices with their respective provenance rather than
silently treating them as the same license.

The public SVN at
[oks003/Model/Febio](https://simtk.org/svn/openknee/oks/oks003/Model/Febio/)
was independently inspected (directory reports revision 3413). The downloaded
files are comparison artifacts under
`Build/open-knee-reference-20261001/upstream-svn-r3413`, not replacements:

- Deck SHA-256: `498e24f8f862f94ea0e27b73fd84ba516d9749629ed53233953ca97d3b6b9c66`.
- Geometry SHA-256: `1e4f4a8c4aacdd701ec17e4761eb398ca9005952922e38fe1119f6d05559ee36`.
- That geometry is 119,017,987 bytes, different from the DOI archive listing.
- That deck uses rigid body 19 for `QSO_With_QAT`, versus 21 in the pinned deck,
  and omits the pinned MCL–meniscus discrete spring.
- Pairing SVN geometry with the pinned deck still fails `MCL_MNS-M_tie` and
  `FMC_To_QAT` resolution. No source-equivalence claim can be made by mixing them.

The [source publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC9832097/)
reports FEBio 2.9 for its simulations. The pinned deck says format **2.5**;
format version is not executable version. No FEBio executable was found on
this host's PATH or in the inspected application/download locations. Exact
solver build, defaults, and independent reference output remain open.

## Required continuation

1. Retrieve the DOI archive's matching geometry and original log, retain hashes
   and download provenance, then rerun reference admission. Inspect the log for
   solver version/settings, termination, and actual achieved load history.
2. Pin the original solver executable or reproducibly built source and run the
   complete source problem in specimen coordinates. Preserve any failed baseline.
3. Lower the complete rigid/joint graph and exact material/prestrain laws into
   existing Matter, with source solver energy/stress/tangent comparisons.
   Continuum fibre restoration must retire the corresponding reduced force.
4. Implement all articular volume mechanics and joint/contact initialization,
   retaining material reference, prepared current coordinates, and prestrain
   separately. Qualify source contact and strict barrier admissibility separately.
5. Add an independently sourced quadriceps-loading experiment at the proximal
   QAT boundary, with unforced PTL tension, full wrench/virtual-work accounting,
   and coupled equilibrium. Passive flexion alone cannot qualify this experiment.
6. Complete swept-contact, refinement, energy/momentum, rollback, and sustained
   loading gates before replacing the local MyoSim prescription in the whole body.

No tissue constitutive code, native force path, or contact kernel was changed
in this admission slice. Existing Matter worktree changes belong to other work
and were preserved. The remaining work above is not complete.

## Verification

Twenty focused tests cover missing attachments/springs/contact pairs,
load-curve and rigid-body references, complete distributed fibre indexing,
unknown-field retention, cylindrical-chain rank, source hash rejection,
byte-preserving freezing, and refusal to overwrite an earlier case. The
existing importer, cartilage-material, and extensor-stack checks are included. The live legacy importer still
reads 16 regions / 248,236 nodes and now retains all nine FEBio sections.
