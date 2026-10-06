# Passive taenia source repair

`numilab_human.resting_taenia_reference.build_candidate` replaces only stable
ID457 (BodyParts3D FJ2569, taenia mesocolica) in the existing NHANAT1 payload.
The source identity, shared torso registration and passive motion owner remain
intact. It adds no mass, forces, physiological compartments or per-step CPU work.
All other record geometry is checked byte-for-byte. A cardiac correction must
be attached by its owner after this source preparation step.

The retained source has two exactly degenerate triangles. Exact coordinate
welding and removal of only those triangles still leaves 311 forbidden self
pairs, including 161 opposite-triangle duplicate components. Exact Boolean
cleanup candidates retained nonmanifold edges and were rejected.

The admitted source candidate uses the existing Blender preparation runtime to
reconstruct this passive muscle band at 0.2 mm voxel resolution and reduce it
to 14,000 inspection triangles. This is offline geometry preparation, not a
volumetric tissue simulation. One diagonal flip and a bounded 4.541 micrometre
vertex adjustment remove the remaining numerical defects. The result is a
closed oriented surface, with minimum triangle altitude 1.766 micrometres and
zero exact self-intersection pairs. Its geometric volume changes from 6.938538
to 6.908539 mL (−0.4324%). No physical body mass is changed.

All source/candidate vertices and triangle centroids were checked in both
directions. Maximum sampled distances are 0.3581 mm source-to-candidate and
0.1371 mm candidate-to-source, below the declared 0.5 mm limit. This is not a
certified global Hausdorff bound. The 0.15/0.20/0.25 mm reconstruction sensitivity
changed geometric volume by −0.2892/−0.4324/−0.6251%; only the conditioned 0.20 mm
candidate received the full exact audit. The shape is explicitly inferred,
not a measurement of a particular adult.

The exact audit covers all 78 other passive surfaces. Every non-colon pair is
clear, including the previously unresolved pancreas comparison. There are
3,877 exact triangle-pair witnesses with six colon components. These are not
silently discarded: the receipt retains every pair and its interpretation.
The source colon envelope includes its muscular wall, while the taenia is a
constituent inspection layer. Band-to-band witnesses occur only 10.8–16.0 mm
from the caudal extent of the source ascending colon; taenia-to-rectum witnesses
occur only 1.02–3.54 mm from the cranial rectal entry. Those localized contacts
are consistent with band convergence and continuation into the rectal wall.
The inference uses the source component identities and actual witness locations;
it does not exempt other crossings on the same surface pair.

Taeniae belong to the colon's longitudinal muscle layer, and join the rectal
longitudinal layer at its entry. See [OpenStax's large-intestine anatomy](https://openstax.org/books/anatomy-and-physiology/pages/23-5-the-small-and-large-intestines),
[rectal anatomy](https://www.ncbi.nlm.nih.gov/books/NBK537245/), and the description
of [taenia convergence at the appendix and rectum](https://pmc.ncbi.nlm.nih.gov/articles/PMC7271214/).
These sources support anatomical interpretation, not the numerical repair
thresholds, measured wall thickness, or clinical validation.

Four source-binding/admission regressions pass on the SSH Mac mini. The actual
compiled payload is retained there at
`/Users/n/numi-human-resting-evidence-20261005/taenia-native-input-092/resting-thorax.nhanatomy`,
SHA-256 `b0efec08f7d62b8559ba5d0ef25ebdaf653f997b2f82546918fa5c0d02f166c5`.
The existing receipt retains source, compiler, audit, shape and interface hashes.
The preparation chain and failed candidates remain in that evidence root,
under numbered steps 076–091. Source licenses and attribution are preserved.

The first native check (run094/audit095) found three self-crossing triangle
pairs after motion, despite the zero-crossing static audit. They share one
local vertex. Its source clearance was 9.297 micrometres, while the native
field moved it 0.655 micrometres through the adjacent face. That failed
candidate and its exact witnesses remain retained.

A further 30.706 micrometre local adjustment establishes a declared 40
micrometre source margin. This is a numerical reference choice, not measured
tissue spacing. Candidate103 remains closed and oriented, retains the 1.766
micrometre minimum altitude, and has geometric volume 6.908534600 mL. Its
sampled shape bounds remain within the original 0.5 mm limit. The existing
compiler admitted it as `taenia-motion-native-input-111/`.

After composition with the bladder correction, native run114 completed six
simulated seconds on the Mac mini. Exact audit115 checks accepted steps 0,
639, 2783 and 2999: every taenia self-count is zero, every non-colon neighbor
is clear, and every retained contact remains within its declared anatomical
interface. Actual native witnesses are mapped barycentrically to source
material coordinates for the caudal-convergence and rectal-entry bounds;
changed triangle pairs are retained rather than silently waived. The bladder
also has zero self-crossings and only its unchanged localized source neck
contact at all four captures. Physiological and surface-observer CSVs are
byte-identical to run094.

The current combined payload is `bladder-cardiac-input-113/resting-thorax.nhanatomy`,
SHA-256 `00128e08b2a83fd6e4b1c3873e50276d1d5574aa32e9fbf891d0221eb15bce60`.
The exact native invocation, unretimed recording, geometry captures, and full
audit remain in `bladder-taenia-native-114/` and `bladder-taenia-native-audit-115/`.
These four captured phases do not qualify unrecorded geometry, remaining
anatomical interfaces, or the required five-minute integrated demonstration.
