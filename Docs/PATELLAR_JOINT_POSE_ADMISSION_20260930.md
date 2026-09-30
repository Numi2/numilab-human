# Patellar clearance pose: whole-body joint admission

The bilateral Open Knee(s) extensor geometry candidate clears its local
patellar/femoral cartilage overlap while preserving the audited bone and
tendon ties. It is **not admitted into the whole-body knee**. The
[source-joint receipt](media/patellar-joint-pose-admission-20260930/receipt.json)
checks the pinned MyoSim `myofullbody` model that owns the displayed and
standing patellar body. It verifies the source archive against the checked-out
joint/equality XML and compiled MuJoCo model, then checks both sides at the
equality-projected neutral pose.

Each `patella_l/r` body has exactly three authored joints: two translations
and one rotation. All three have active joint equalities prescribing their
coordinates as fourth-order polynomials of that side's knee angle. At a fixed
knee angle, those source laws provide **zero independent exact patellar pose
coordinates**. The receipt records the coefficients and their prescribed
values at 0, 0.9 and 2.0 rad. It also binds the unchanged 20 µm Open Knee(s)
clearance vector and coherent-extensor receipt on each side. Moving the
MyoSim patella independently to implement that vector would introduce an
equality residual or require changing the source law. The MuJoCo equalities
have finite solver compliance, so this is an exact authored-kinematics gate,
not a stiffness or loaded-force measurement.

Open Knee(s) and MyoSim are separate source models with different frames;
this audit intentionally does not treat the Open Knee translation components
as MyoSim world coordinates. No validated mapping or loaded, coupled
bone/cartilage/tendon transaction currently admits the candidate. The
original source rest pose still has 18 cartilage-surface crossings. A native
initial-contact policy or source-backed registration/joint calibration must
resolve those crossings while preserving joint equalities, attachments,
contact force and energy. The 20 µm geometry result stays unadopted until
that owner passes these checks. This receipt makes no clinical-anatomy or
standing-simulation qualification claim.

The subsequent [bone-tie-fixed cartilage field](PATELLAR_CARTILAGE_TIE_CLEARANCE_20260930.md)
avoids an independent rigid-joint translation by leaving the patella and its
bone tie fixed. Its one-step native check is still a PTC/FMC-only diagnostic;
it has not admitted a new whole-body pose or loaded contact law.

Reproduce the CPU source audit:

```sh
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/audit_patellar_joint_pose_admission.py
```
