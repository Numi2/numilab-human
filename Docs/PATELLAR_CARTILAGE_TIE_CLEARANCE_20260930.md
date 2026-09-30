# Bone-tie-fixed patellar cartilage clearance

The source Open Knee(s) patellar/femoral cartilage surfaces have 18 crossing
triangle pairs on each side. Moving the entire patellar extensor stack by
20 µm clears those crossings geometrically, but that translation has no
independent whole-body patellar joint coordinate. Moving cartilage alone
uniformly would also detach it from the patellar bone.

The [bilateral geometry receipt](media/patellar-cartilage-tie-clearance-20260930/receipt.json)
tests a local cartilage field instead. It keeps all **4,592 PTC/PTB bone-tie
nodes fixed** and tapers the existing side-specific 20 µm clearance direction
through the cartilage volume. The field is one at 5,681 articular-surface
vertices. The other two articular vertices are also bone-tie seam vertices
and stay fixed. Patellar bone, QAT, PTL, the source reference mesh and the
MyoSim patellar joint law do not move.

Both compiled sides have zero exact PTC/FMC articular or full-boundary face
crossings, zero exact tetrahedral boundary contacts or interior overlaps, and
no PTC surface self-intersection. All **121,105 PTC tetrahedra per side** keep
positive orientation. The minimum/maximum deformation-volume ratios are
**0.948323–1.025736**, and the cartilage-to-bone tie gap changes by exactly
zero after Float32 rounding. This is a geometric/current-position candidate:
the strain relative to the source reference is not a calibrated prestress.

The [Apple M4 native diagnostic](media/patellar-cartilage-tie-native-20260930/receipt.json)
uses the existing 50,991-node, 208,177-tetrahedron PTC/FMC Matter step with
the authored tie-node set fixed. Numi Lab `coupled` commit `5b51771` adds the
explicit fixed-node input to that probe. With the candidate current positions,
both sides accept one **1 µs** microstep; all 4,592 tie nodes remain bitwise
fixed, and both accepted cartilage volumes remain exactly disjoint. The left
accepted state replays byte for byte. With the original crossing source pose,
both sides reject with status 6, zero accepted microsteps and bitwise rollback.
Without the fixed-node input, all 4,592 tie nodes drift after one accepted
microstep, up to **7.477 µm** left and **7.377 µm** right.

The candidate's contact-on and contact-off accepted position streams are
identical on each side, with **zero active deformable-contact histories**.
The step therefore demonstrates strain relaxation with a static bone tie,
not a loaded patellofemoral contact force. Its material is a synthetic probe
material. The fixed nodes do not yet return reactions to the articulated
patella, and QAT/PTL are absent from this Matter scene. The field remains
**unadopted** pending source-frame registration, initial strain/stress and
material calibration, two-way bone and tendon reaction, contact force/energy
closure, and sustained flexion. It does not establish clinical anatomy or a
corrected whole-body standing simulation.

Reproduce the geometry and native receipts from the pinned sources and current
local Numi Lab build:

```sh
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/audit_patellar_cartilage_tie_clearance.py
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/audit_patellar_cartilage_tie_native.py
```
