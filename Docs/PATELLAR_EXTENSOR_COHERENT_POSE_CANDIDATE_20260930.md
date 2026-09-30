# Coherent extensor geometry for the unadopted cartilage clearance pose

The exact [PTC/FMC solid-separation gate](PATELLOFEMORAL_TETRAHEDRAL_SEPARATION_20260930.md)
passes for a 20 µm patellar-cartilage translation, but moving cartilage alone
would separate it from its patellar bone tie. The [extensor-pose receipt](media/patellar-extensor-pose-candidate-20260930/receipt.json)
tests a broader, **unadopted** field on both pinned `NHKNEE1` sides:

- Translate patellar bone (`PTB`), patellar cartilage (`PTC`) and quadriceps
  tendon (`QAT`) together by the existing side-specific clearance vector.
- Taper patellar tendon (`PTL`) displacement from that vector at its patellar
  tie to zero at its tibial tie, without moving the tibia.

The candidate's PTC Float32 positions match the native cleared-contact input
bit for bit, so the prior exact PTC/FMC solid-separation result applies. All
four authored tie-set pairs retain their bidirectional nearest-node gaps
exactly after Float32 rounding: PTC–PTB, QAT–PTB, PTL–PTB and PTL–TBB. The
553 PTL tibial-tie nodes do not move. The PTL volume ratio stays between
**0.9980719** and **1.0067979** on both sides. Every candidate PTC, QAT and
PTL tetrahedron has positive exact oriented volume, every PTB triangle is
nondegenerate and unreversed, and all four changed source surfaces remain
closed and free of exact self-intersections on both sides.

This closes the *geometric* bone/cartilage/tendon tie-preservation and local
element-inversion questions for the candidate. It does not adopt a new patella
pose. The rigid patella joint frame and equality program are unchanged, QAT's
proximal muscle attachment is not qualified, and PTL's taper is an authored
geometric construction rather than an equilibrated material state. No native
step includes the bone and both tendons in this candidate; contact pressure,
reaction, work, sustained loading, source anatomy and clinical validity remain
open. The original source pose still has 18 PTC/FMC surface crossings.

Reproduce from the pinned Open Knee(s) source, bilateral compiled payloads,
native cartilage inputs and earlier separation receipt:

```sh
PYTHONPATH=src:Sources/myosim/checkout:. OPENBLAS_NUM_THREADS=1 \
  .venv-mujoco312/bin/python tools/audit_patellar_extensor_pose_candidate.py
```
