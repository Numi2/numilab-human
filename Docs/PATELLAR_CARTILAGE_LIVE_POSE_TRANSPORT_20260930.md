# Patellar cartilage in the live Human pose frame

The [source-bound pose-transport receipt](media/patellar-cartilage-live-pose-transport-20260930/receipt.json)
finds a frame mismatch before the local patellar cartilage clearance candidate
can enter the live Human/Matter knee. `NHKNEE1` was compiled at MyoSim's literal
`qpos0`, but the six-tissue live runner treats its `restWorld` nodes as being
in the equality-projected neutral body frame. At that projected pose, the
old payload's patellar cartilage visual is **53.45 mm** from the mechanical
rest position on both sides. The femoral cartilage agrees within **0.07 µm**.
The apparent separation of the old visual and the mechanical cartilage is
therefore not evidence that the live knee is anatomically coherent.

The Open Knee compiler now offers `--projected-visual-frame` for an **opt-in,
unadopted candidate**. It derives visual and rigid-attachment local points
from the equality-projected MyoSim inertial-body frames. The candidate changes
only these node-local payload bytes; the source `restWorld` coordinates,
tetrahedra, surface data, material data, and joint law are byte-identical.
Every region's visual-to-rest residual at projected neutral is below
**0.04 µm**. A default-mode recompilation still produces the exact existing
left `NHKNEE1` payload SHA-256, so existing evidence stays reproducible.

The receipt also runs the same geodesic inverse-distance continuum-map rule
used by the native PTL path as a **Python preflight**. Starting the patellar
tendon from the old default frame and transporting it into projected neutral
inverts one of 35,616 tets on each side (minimum Jacobian about -0.21). With
the projected frame as its reference, minimum PTL Jacobian is 1.0 at neutral,
0.921/0.905 at 0.1 rad, and 0.773/0.367 at 0.9 rad (left/right). This is
not a native acceptance or force certificate.

The corrected frame exposes the next geometric failure. The tie-fixed 20 µm
PTC field has zero exact PTC/FMC articular crossings at projected neutral,
but **164 crossing triangle pairs per side at 0.1 rad**, and **416 left / 414
right at 0.9 rad** under the source patellar equality law. Those are exact
intersection predicates on Float32 positions transported through the source
inertial-body frames. The bilateral field and projected-frame payload remain
**unadopted**. A native loaded step, contact-force and energy closure,
calibrated source kinematics, and sustained flexion are still required before
the whole-body knee can be claimed anatomically or mechanically corrected.

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
