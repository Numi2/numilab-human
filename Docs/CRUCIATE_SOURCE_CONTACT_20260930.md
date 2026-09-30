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
transfer**: applied PTL force 2.26 N versus patellar reaction 8.68 N. The
variant releasing the three patellar equalities also completes one step but
has **zero quadriceps transfer**, so it fails before loaded-motion admission.
Both app invocations exit nonzero. Neither confirms correct patellar
placement, force closure, sustained stance, or clinical anatomy.

The next implementation must reconcile the source's initially intersecting
contact surfaces with a source-consistent contact or equilibrated
initialization policy across all affected pairs, then close PTL/quadriceps
force transfer and rerun loaded flexion, articular clearance, energy,
replay, and rollback gates. A localized ACL shift cannot serve as the final
whole-knee solution.
