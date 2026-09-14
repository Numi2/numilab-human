# Muscle surface geometry audit

`numilab_human.muscle_surface_geometry_audit` replays the exact BodyParts3D
`isa_BP3D_4.0_obj_99.zip` members behind the NHTISS4 whole-body muscle/tendon
manifest. It checks each member hash and source count, applies the same exact
authored-coordinate quotient used by the organ geometry compiler, recomputes
surface area, and records topology defects. It is available as:

```sh
PYTHONPATH=src python -m numilab_human.muscle_surface_geometry_audit \
  --output Docs/media/muscle-surface-geometry-audit-20260914/receipt-v1.json
```

The audit covers 148 muscle surfaces and two tendon surfaces. The source
recomputation has 316,420 quotient vertices, 631,464 triangles, 3.2682904751
m² of surface area, 60 single closed components, six closed multi-component
surfaces, and 84 topology-defective surfaces. The 60 single closed components
have a combined algebraic volume candidate of 0.006471304533 m³. These are
source geometry candidates only; the receipt assigns no physical volume,
mechanical mass, material, active-force, support, or subject-calibration owner.

The six multi-component and 84 defective surfaces remain visible as failed
volume admissions. This prevents the visual NHTISS4 payload from becoming a
silent muscle or tendon continuum and gives the later tissue-mass and active
force work a measured source gate to repair.

The immutable receipt is `media/muscle-surface-geometry-audit-20260914/receipt-v1.json`;
its SHA-256 is recorded in the adjacent `SHA256SUMS` file. The cross-domain
receipt `media/body-composition-integration-20260914/receipt-v7.json` binds the
audit hash, surface-area candidate, algebraic volume candidate count, and the
zero-owner assertions while keeping `skeletal_muscle_tissue_volume` false.
