# Supported initialization and skin/fat transfer on Apple M6

The authoritative native full-resolution control retains 1,448,804 repaired
skin/fat cells and 13.939678963460478 kg of mapped skin/fat mass. The offline
volume gates passed after repairing 125 inversions and 154 shape failures.
The native registration admitted all 34 ligament rows, including both shoulders.
This experiment keeps those sources and the existing Matter bed owner.

## Muscle recruitment experiment

Using the full repaired volume and the exact control launch, a 0.10 MyoSim
activation cap reduced the first-root peak generalized acceleration from
580.59 to 198.53. The accepted right/left ankle accelerations changed from
−580.59/−572.33 to −11.92/−14.76 rad/s². The accepted knee accelerations went
from 23.30/−4.50 to 198.53/151.85 rad/s². The analytical knee torque residuals
increased as well. Both runs reported zero reaction from the bed.

The trial advanced 0.25 ms in 219.32 s of body runtime and 360.70 s from
process launch. These correspond to 1.14e−6× and 6.93e−7×, respectively.
Body and physiology accepted clocks matched without root assistance.
Pleural pressure remained −490.332489014 Pa throughout this prefix.
The run had no native presentation qualification. Recruitment remains a
tested release diagnostic and is unsuitable as a supported start state.

## Native parent and offline transfer

The same-source coarse skin/fat parent has 7,233 nodes and 28,632 tetrahedra.
The fine launch identifies this exact parent and its topology. The first
coarse candidate with the fine streamed tangent configuration was inadmissible.
Removing the fine regional cache still failed the first step under the rest
of the fine solver settings. The authored coarse settings accepted the first
two roots and rejected the third with a nonlinear residual of 1.15315e−5
against 1.0e−5. Reusing the fine mixed-precision contact settings gave the
same accepted prefix and rejection. Neither case changed physical tolerances.

The exact two-root capture completed 31.25 μs with zero bed reaction.
The endpoint observer provided matching global IDs, local IDs, initial
Float32 coordinates and masses for all 7,233 parent rows. The transfer code
uses the typed, pinned parent ancestry and applies one shared displacement
to each of the 118,196 skin/fat interface nodes. It retains the original
reference mesh and measures current and reference-relative determinant
ratios independently. No trial geometry was installed in the native scene.

The identity transfer preserves all 311,020 fine nodes and 1,448,804 cells.
A manufactured 1 mm translation conserves reference-relative determinants
to within 5e−11. Absolute Float32 output from the same upward translation
pushes source cell 6873 to 0.2492 of its reference determinant. The downward
translation leaves 19 outer points inside the unchanged bed and is refused.
The real two-root parent transfer reaches 62.62 μm displacement, inverts
367 fine cells, and puts 492 cells below the motion floor. The lowest
reference-relative determinant ratio is −224.09, so the failure persists
without absolute-coordinate rounding. The worst cell spans three parent
tetrahedra. Transporting the registered residual offsets with the parent
gradients is affine-consistent, but worsened this endpoint to 1,951
inversions and 2,577 cells below the motion floor. These receipts rule out
direct transfer of this parent trajectory as the fine starting state.

The archived workspace retains the native launches, exact asset and binary
pins, successful and rejected native logs, and compact transfer witnesses.
Closed replay packs were transparently compressed with matching before/after
hashes. This experiment does not qualify equilibrium, crossings through
motion, varying pleural pressure, or full-scene throughput.

Run the diagnostics on SSH Numi using the paths pinned by the retained
launches. `run_recruitment.py` reproduces the activation comparison,
`run_same_source_parent.py --steps 2` captures the native parent endpoint,
`prolong_accepted_parent.py same-source-parent-2` transfers its positions
offline, and `verify_prolongation.py` runs the manufactured translations.

## Third-root rejection isolated

The compensated vascular state alone left vascular residual 3.8673e−5.
The existing fixed-geometry vascular predictor reduced that block to
3.3415e−9 against the unchanged 1e−5 limit. It uses that result as a guess
for the existing coupled solve. The third step still failed the mechanical
certificate at 1.5746e−3 against the 1.25e−5 domain threshold. Only the same
two roots were published. The equivalent direct preconditioner dispatch was
required by the runtime's coarse predictor admission.

A paired trace shows that the fresh-direction reset works on the Metal
buffer. Newton iterations 1 and 2 started with a shared bound of one,
while the subsequent tissue calculation independently selected the same
small bounds as the original. The applied fraction was 3.7253e−10 in the
last iteration in both trials. The parallel constitutive trace maps the
upper-determinant witnesses to source parent fat cells 26104 and 26201,
which share three nodes. The two-root accepted native determinants are
2.2095 and 2.4461. One free node moved about 20.52 μm while its neighbors
moved less than 0.69 μm. The reference edge bases have condition numbers
2,456 and 1,424. This isolates the next geometry work to a real material
limit at the hand and supplies the exact attachment-free stencil. The
predictor and reset have no production promotion from this aborted run.
