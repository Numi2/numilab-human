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

Numi Lab `coupled` commit `dbe3df2` also captures the raw **815,856-byte**
GPU constraint-reaction stream for each accepted step. The Human auditor
independently reduces those Float32 node forces over the exact PTC/PTB tie set.
The synthetic-material candidate produces a **639.078 N** resultant and
**0.952 N·m** moment about the tie centroid on the left; the mirrored right
produces **638.876 N** and **0.951 N·m**. Left/right force mirror mismatch is
0.202 N. The norm of reaction plus accepted momentum rate is
**4.239 N** left and **7.856 N** right (0.66%/1.23% of resultant), so exact
force closure has not been established. The raw reaction stream replays
bitwise on the left and matches between contact-on/off controls on both sides.

The candidate's contact-on and contact-off accepted position streams are
identical on each side, with **zero active deformable-contact histories**.
The step therefore demonstrates strain relaxation with a static bone tie,
not a loaded patellofemoral contact force. Its material is a synthetic probe
material. The measured tie reaction is not yet applied to the articulated
patella, and QAT/PTL are absent from this Matter scene. The field remains
**unadopted** pending source-frame registration, initial strain/stress and
material calibration, two-way bone and tendon reaction, contact force/energy
closure, and sustained flexion. It does not establish clinical anatomy or a
corrected whole-body standing simulation.

The later [live pose-transport audit](PATELLAR_CARTILAGE_LIVE_POSE_TRANSPORT_20260930.md)
measures the raw-payload frame mismatch directly: its old patellar visual
coordinates place cartilage about 53.45 mm from `restWorld` at projected
neutral. The live runner already reconstructs that visual from `restWorld`.
An opt-in projected-frame payload aligns the raw coordinates, but exact
PTC/FMC crossings recur under the source flexion law. Neither candidate is
adopted.

Reproduce the geometry and native receipts from the pinned sources and current
local Numi Lab build:

```sh
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/audit_patellar_cartilage_tie_clearance.py
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/audit_patellar_cartilage_tie_native.py
```
