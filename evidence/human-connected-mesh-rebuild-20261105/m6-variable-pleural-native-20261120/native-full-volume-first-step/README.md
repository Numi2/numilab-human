# First physical step with the repaired whole-body skin and fat

The source audit and repair covered 125 inverted and 154 low-quality cells.
The final emitted volume passed both the normalized shape and the absolute
reference-volume gates across 1,448,804 cells. Its changed layer-boundary
faces had zero reference-pose crossings. The replaced native skin/fat
volume retained 13.939678963460478 kg from its captured runtime parent.
All 34 source ligament rows were admitted, including the two reattached
shoulders.

The full scene ran on SSH Numi's Apple M6 with the registered tissue,
sole FEM bed, abdominal aortic wall, and variable pleural-pressure
extension. The independent Metal fixture verified the existing pressure
ramp and two different pleural outputs over all 21 compartment rows.
The abdominal aortic boundary and all input buffers were byte-identical.

Runs 005 and 006 held the same 15.625 microsecond diagnostic root and
the same physical convergence gates. The baseline took 484.20 seconds
from native launch and the exact contact cache took 445.21 seconds.
Both rejected the first step. Their native failure receipts are
byte-identical. The exact storage encoding retained all 15,449,893
contact rows and reduced the allocation at the adapter by 1,032,142,848
bytes. Across its 24 Newton iterations, the median CPU arrival intervals
between GPU stage signals were 0.208 seconds for assembly, 2.847 seconds
for the linear solve and 9.823 seconds for globalization. These are
device progress intervals in an instrumented trial and include observer
delivery. They are representative of this rejected root only.

The baseline exhausted all 24 linear solve budgets. Mechanical
residual 0.71025604 repeated at each Newton assembly and the freshly
assembled mechanical residual was 0.710255623. The terminal vascular
certificate reported 1.226533277076669e-05 against 9.999999747378752e-06.
The common cardiac geometry candidate remained at its initialization
coordinates. The coupled solver published no physical time.

## Contact globalization and load-continuation trials

Runs 007 and 008 used two Newton passes as diagnostics. The parallel
constitutive bounds reduced the measured globalization interval to
4.100 seconds. Adding the tiled deformable limiter reduced it to
0.735 seconds with the same initial constitutive and contact bounds,
128 Krylov columns over two passes, and a byte-identical native
failure receipt. This is a stage-speed measurement on a rejected
step. The constitutive safeguard selected 0.001953125. The tiled
contact limiter selected 0.00000355235238 based on a per-feature
barrier at contact 1208222. Its swept test certified the incoming
chord clear. The vascular limiter then further reduced the fraction
to 0.0000000000135511489.

The witness audit maps contacts 1208222 and 1208247 to the same
point–triangle feature reached from two distinct primitive pairs.
The point and its initial closest corner are 8.358 micrometres apart
and share TetGen cell 530625. The four support nodes do not all
belong to one tetrahedron. The nearby rest-tet edge-basis condition
number reaches 17,292 even though its signed volume and the required
normalized determinant are positive. This is a local geometry
conditioning witness, not a measured condition number of the
assembled operator. A node in the proposed correction moves about
42.7 micrometres in the chord used for the first bounds check. The
audit preserves the contact rows and identifies no crossing at
these particular CCD witnesses.

Runs 009 and 010 attempted the existing compensated vascular
predictor. Both ended during solver admission. The existing prefix
command-buffer preconditioner excludes the predictor. Disabling it
also requires disabling the dependent flat-column and aggregate
command-buffer paths. The later failure was surfaced with a
respiratory configuration error after the runtime returned a
success flag early.

Run 011 applied the existing accepted-step gravity and vascular
traction continuation. It left the circulation, mass, contact
and final certificates active. Both load ramps started at zero
and would reach their full authored loads on the second accepted
step. There were zero accepted steps. The vascular limiter no
longer shortened the proposed correction, but contact 1208222
still restricted the first Newton fraction to 0.00000354738631.
The terminal mechanical residual was 0.710250795. The experiment
shows that reduced initial applied loads are insufficient to
clear the observed blocker.

All zero-crossing checks in this packet concern the reference
skin/fat layer or identify an individual CCD witness. The existing
static geometry and registration results stand. No whole-body
motion check, settled-state equilibrium or nonzero integrated
real-time factor was obtained.
