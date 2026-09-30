# Open Knee initial contact and bounded native diagnostic — 2026-09-30

The pinned left Open Knee source mesh contains **84 strict triangle crossings**
on its authored `PCL_To_ACL` sliding contact surfaces. Their adjacent PCL and
ACL tetrahedra overlap in all 84 cases. Both native step-zero failure faces
map exactly to those source contact surfaces and cross in the source rest
geometry. The contact is therefore not an invented broadphase pair, and
simply disabling it would discard a source-authored interface. The
source-bound verifier is `tools/audit_open_knee_cruciate_source_contact.py`,
with results in [`receipt.json`](media/cruciate-source-contact-20260930/receipt.json).

The wider source audit found **8 of 19** authored contact pairs have initial
strict surface crossings, totaling **2,190** crossing face pairs. This is a
general source initialization issue; resolving the PCL–ACL pair alone does
not establish a nonintersecting whole knee. The PCL–ACL crossing patch has
37 PCL and 46 ACL surface nodes, none rigidly tied to bone.

An opt-in, left-neutral-only native diagnostic moves free ACL nodes locally
toward source-world +x by at most 0.5 mm. The original source coordinates
remain the FEM material reference; all rigidly attached nodes stay fixed.
The independent [`initialization-candidate.json`](media/cruciate-source-contact-20260930/initialization-candidate.json)
checks 4,448 moved nodes, zero remaining crossings in the three named ACL
contact pairs, and no inverted ACL tetrahedra (minimum source-to-candidate
volume ratio 0.8485). This is a local initialization candidate, not a
source-authored unloaded shape or an anatomy correction.

The [matched native receipt](media/cruciate-source-contact-20260930/native-initialization-receipt.json)
binds the final binary and payload hashes to three 1 µs runs. The untouched
source is rejected by deformable contact before step zero completes. Both
opt-in candidate variants complete one coupled step. The constrained
variant verifies replay and rollback but **fails patellar-tendon force
transfer admission**: applied PTL force 2.26 N versus total patellar reaction
8.68 N. The
variant releasing the three patellar equalities also completes one step but
has **zero quadriceps transfer**, so it fails before loaded-motion admission.
Both app invocations exit nonzero. Neither confirms correct patellar
placement, force closure, sustained stance, or clinical anatomy.

The follow-up [PTL force-accounting receipt](media/cruciate-source-contact-20260930/ptl-force-accounting-receipt.json)
binds the diagnostic to Matter `02de76c`, its final binary, the same payloads,
and one native 1 µs step. The active PTL load is assembled (force-assembly
gate passes), with equal and opposite 2.26 N patch loads. The total fixed-node
reaction includes that load plus internal, contact, gravity, and inertial
terms. Subtracting the known active couple leaves 10.80 N at the patellar
attachment and 5.51 N at the tibial attachment; the combined PTL attachment
reaction is 10.21 N. These residuals are **not** isolated passive forces or
proof of a new anatomical defect. The existing PTL gate still rejects the
run, and the app exits nonzero.

The subsequent [one-step momentum receipt](media/cruciate-source-contact-20260930/ptl-momentum-one-step-receipt.json)
binds Matter `c7464bf` and the final binary to the same inputs. PTL enthesis
centroids separate by only 27 nm in the accepted step, and the independently
evaluated reduced fiber law contributes 0.000201 N. The 4.423 g tendon
continuum changes momentum at a vector rate almost opposite to the combined
10.21 N attachment reaction: their vector sum has magnitude 0.0714 N (0.70%
of the attachment resultant). This attributes the *combined* PTL attachment
reaction to one-step continuum momentum change within that residual; it does
not isolate the matrix force at either patch. The reduced fiber tension is
negligible at this step. Comparing each *total* attachment reaction
with the 2.26 N active patch load is not a valid stand-alone force-transfer
test. At that revision the physical gate remained unchanged and rejected the
run. A replacement needed accepted-step vector impulse closure and separate
contact and energy accounting.

Matter `a334f99` now accumulates PTL attachment impulse only after each Human
step is accepted, with a step-index guard against replay double counting. Its
PTL gate checks active load assembly and impulse-versus-continuum-momentum
closure within 2% of the sum of accepted reaction-impulse magnitudes. The
[matched native trajectory receipt](media/cruciate-source-contact-20260930/ptl-trajectory-native-receipt.json)
binds the final binary, shader, source, payloads, commands, and four logs. The
local ACL diagnostic completes one and two 1 µs steps with 0.70% and 1.10%
PTL impulse residual respectively, and verifies bitwise replay and rollback.
Omitting the second accepted impulse in a negative control gives 74% error.
These short runs remain explicitly marked `unqualified_local_acl_initialization`.

The untouched source still rejects step zero with zero accepted PTL impulse.
With the local ACL candidate, the eight-step request accepts five steps, then
rejects the sixth at PCL–ACL contact; the PTL audit retains exactly five
accepted impulses. The independent [contact witness](media/cruciate-source-contact-20260930/ptl-trajectory-contact-witness.json)
maps the reported pair to pinned PCL face 3525 and ACL face 6832. These faces
do not strictly cross in the source, candidate, or native step-start geometry,
but strictly cross at the rejected step finish after at most 3.45 µm of
reported vertex motion. The local ACL shift therefore does not give a stable
contact trajectory. The solver's rejection remains authoritative. A source- and binary-bound [contact-pass receipt](media/cruciate-source-contact-20260930/contact-pass-stage-receipt.json) now identifies the first strict predicted-finish crossing at nonlinear contact pass 2 (zero-based solver iteration 1). This is after the first Newton update and before the next contact force evaluation. The current selected-feature line search has not certified all surface-pair motion; the fail-closed rejection is retained.

The next implementation must reconcile the source's initially intersecting
contact surfaces with a source-consistent contact or equilibrated
initialization policy across all affected pairs. The accepted-step PTL impulse
gate is now measured for two microseconds, but sustained PTL/quadriceps force
transfer, loaded flexion, patellar clearance, energy, replay, and rollback
still need full-horizon evidence. A localized ACL shift cannot serve as the
final whole-knee solution. The next contact-solver check must establish whether
the crossing pair was in the preceding correction's broadphase candidate set;
if absent, the broadphase must cover the swept Newton correction before a
triangle-pair CCD bound can certify that correction.
