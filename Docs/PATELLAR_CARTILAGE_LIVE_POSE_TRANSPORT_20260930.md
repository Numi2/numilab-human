# Patellar cartilage in the live Human pose frame

The [source-bound pose-transport receipt](media/patellar-cartilage-live-pose-transport-20260930/receipt.json)
finds a raw-payload frame mismatch before the local patellar cartilage
clearance candidate can enter the live Human/Matter knee. `NHKNEE1`
`visualLocal` coordinates were compiled against MyoSim's literal `qpos0`,
while `restWorld` is the source reference used at equality-projected neutral.
At that pose, the **raw payload** patellar cartilage visual is **53.45 mm**
from `restWorld` on both sides. The femoral side agrees within **0.07 µm**.
The existing **live** runner already reconstructs its non-FEM visuals from
`restWorld` in the projected body frame, so its displayed cartilage does
not inherit that offset. The raw visual separation is not evidence of live
cartilage clearance.

The Open Knee compiler now offers `--projected-visual-frame` for an **opt-in,
unadopted candidate**. It derives raw visual and rigid-attachment local points
from the equality-projected MyoSim inertial-body frames, matching the frame
the live runner already reconstructs. The candidate changes only these
node-local payload bytes; the source `restWorld` coordinates,
tetrahedra, surface data, material data, and joint law are byte-identical.
Every region's visual-to-rest residual at projected neutral is below
**0.04 µm**. A default-mode recompilation still produces the exact existing
left `NHKNEE1` payload SHA-256, so existing evidence stays reproducible.

The receipt also runs the same geodesic inverse-distance continuum-map rule
used by the native PTL path as a **Python preflight**. A hypothetical map
starting the patellar tendon from the old default body frame and transporting
it into projected neutral inverts one of 35,616 tets on each side (minimum
Jacobian about -0.21). **The live runner does not use that map**: it starts
from projected rest. With that reference, minimum PTL Jacobian is 1.0 at neutral,
0.921/0.905 at 0.1 rad, and 0.773/0.367 at 0.9 rad (left/right). This is
not a native acceptance or force certificate.

The projected frame exposes the next geometric failure. The tie-fixed 20 µm
PTC field has zero exact PTC/FMC articular crossings at projected neutral,
but **164 crossing triangle pairs per side at 0.1 rad**, and **416 left / 414
right at 0.9 rad** under the source patellar equality law. Those are exact
intersection predicates on Float32 positions transported through the source
inertial-body frames. The bilateral field and projected-frame payload remain
**unadopted**. A native loaded step, contact-force and energy closure,
calibrated source kinematics, and sustained flexion are still required before
the whole-body knee can be claimed anatomically or mechanically corrected.

The [source-joint clearance envelope](media/patellar-joint-clearance-envelope-20260930/receipt.json)
tests bounded changes to MyoSim patellar `translation1` while holding the
knee angle and other source coordinates fixed. Every nonzero change violates
the active authored equality, so it is a diagnostic rather than a proposed
joint law. At **0.1 rad**, a **6 mm** shift in the posterior direction clears
the exact articular crossings on both sides. The Python PTL continuum-map
preflight remains positive (minimum Jacobian **0.733 left / 0.671 right**),
and the patellar bone remains roughly **39 mm anterior** to the femoral knee
origin. At **0.9 rad**, none of the tested shifts from -8 to +1 mm clears the
crossings. The -8 mm position has **455 left / 451 right crossings** and
places the patellar bone about **20 mm anterior**; the right PTL map also has
two tetrahedra below its 0.05 Jacobian gate. The registration's 25 mm
anterior-offset floor was established at neutral, so the flexed offsets are
context rather than a validated flexion threshold. A new source-bound
patellar motion law or registration needs more than a one-coordinate shift.

Reproduce the compiler candidate and geometry preflight with the pinned
MyoSim/MuJoCo environment:

```sh
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python -m numilab_human.open_knee \
  --sources Sources --open-knee Sources/open-knee-oks003 \
  --registration Docs/media/skin-shell-candidate-20260914/registration.json \
  --output Build/open-knee-projected-visual-frame-20260930/left \
  --side left --projected-visual-frame
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python -m numilab_human.open_knee \
  --sources Sources --open-knee Sources/open-knee-oks003 \
  --registration Docs/media/skin-shell-candidate-20260914/registration.json \
  --output Build/open-knee-projected-visual-frame-20260930/right \
  --side right --projected-visual-frame
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/audit_patellar_cartilage_live_pose_transport.py
```
