# Muscle geometric volume candidate

The `muscle-geometric-volume-candidate` compiler binds the 60 single-closed
BodyParts3D muscle surfaces from the geometry audit to immutable source-member
IDs. It is available through:

```sh
PYTHONPATH=src python -m numilab_human.muscle_volume_owner_candidate \
  --output Docs/media/muscle-geometric-volume-candidate-20260914/receipt-v1.json
```

The candidate covers 60 of 148 muscle surfaces (40.5405%) and reproduces the
audit's `0.006471304532959316 m^3` algebraic volume total. Each admitted row
has a stable `muscle-volume:bodyparts3d:<member>` identity and the upstream
member hash. The six closed multi-component surfaces and 82 topology-defective
muscle surfaces remain excluded.

This closes a source-bound geometric handoff only. The rows are not watertight
FEM volumes and do not assign density, mechanical mass, material, active-force
transfer, activation calibration, subject calibration, or standing/walking
behavior. Tendon, fat, and skin volume owners remain absent. The cross-domain
integration receipt v8 records this candidate while keeping all physical-owner
and calibration gates false.

Receipt: `media/muscle-geometric-volume-candidate-20260914/receipt-v1.json`.
