# Patellofemoral initial-pose candidate - 30 September 2026

The [exact source-bound audit](media/patellofemoral-pose-clearance-20260930/receipt.json)
tests a small bilateral pose change against the complete published PTC and
FMC boundary-face sets. Each side starts with **18 crossing face pairs**.
Translating the patellar-cartilage positions **20 µm opposite the mean outward
normal of the crossing patch**, then rounding to the native Float32 position
format, leaves **zero crossing or point-contact face pairs** in both knees.
The check covers 21,756 PTC and 40,824 FMC boundary faces per side. Source
and compiled payloads were read and hash-checked; neither was modified.

This is a geometric *candidate*, not a corrected knee. Only PTC positions were
translated in the audit. A usable initial pose must move the patellar bone and
cartilage coherently, retain their tie, and account for the quadriceps and
patellar-tendon attachments, source rest configuration, stress, reactions and
energy. It must then execute in the native coupled transaction under load.

The broader topology check also found a source defect that limits a volume
claim: FMC `All_Faces` has one vertex with **two separate surface-link fans**
on each side (source node `233523`, compiled node `233522`). It has no boundary
edge or nonmanifold edge, but is not a closed vertex-manifold boundary. PTC
passes the corresponding closed oriented manifold check after unused interior
nodes are excluded. Zero interdomain face crossings therefore does not yet
prove the cartilage volumes are disjoint; the femoral boundary topology needs
source-aware resolution or a justified alternative volume test.

The [native contact guard](PATELLOFEMORAL_NATIVE_CROSSING_STEP_20260930.md)
continues to reject the original intersecting two-cell input with rollback.
No loaded-contact, clinical-anatomy or whole-knee qualification follows from
this pose candidate.

Reproduce the exact geometry and topology audit:

```sh
cd /Users/home/numilab-human
PYTHONPATH=src:Sources/myosim/checkout OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python -m tools.audit_patellofemoral_pose_clearance
```
