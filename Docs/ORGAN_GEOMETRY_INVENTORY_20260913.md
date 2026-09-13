# BodyParts3D organ and vessel geometry inventory, 13 September 2026

The new `organ-geometry-inventory` owner resolves the 18 regions declared by
`config/physiology-organ-network-template.v1.json` against the pinned
BodyParts3D 4.0 `part_of` hierarchy and the exact
`partof_BP3D_4.0_obj_99.zip` archive. It records each source OBJ member's
bytes, hash, authored bounds, raw topology, and the exact decimal-coordinate
seam quotient used by the cardiac cavity importer. The resulting immutable
[inventory](media/organ-geometry-inventory-20260913/inventory.json) is the
source-data receipt for this increment.

The inventory contains 378 unique source members across 386 declared region
memberships. Eight memberships intentionally share a source member because the
BodyParts3D hierarchy overlaps at anatomical boundaries. Exact seam
identification leaves 371 closed oriented manifold candidates. Seven source
members remain defective after that quotient: `FJ2434` in the right ventricle,
`FJ2928` in the left lung, and `FJ2404`, `FJ2405`, `FJ2409`, `FJ2820`, and
`FJ2821` in the liver. Those records and their hashes are retained in the
inventory; no faces or vertices are repaired.

The command is:

```sh
PYTHONPATH=src python3 -m numilab_human.cli organ-geometry-inventory \
  --sources Sources \
  --output Docs/media/organ-geometry-inventory-20260913/inventory.json
```

The owner verifies source membership and archive identity, but it does not
perform a watertight multi-member union, self-intersection or interdomain
overlap test, body-frame registration, material or density assignment, blood
mass assignment, or physical stepping. A closed member is therefore a source
geometry candidate, not an admitted organ or vessel mechanical field. The
cardiac wall's separate CT-derived defects, unloaded-state and material gaps
remain unchanged.

This increment advances `organs.data` through a reproducible source inventory;
`Organs, vessels, and nerves` remains **open** until registered volumetric or
tubular fields, calibrated materials and boundaries, conservative coupling,
and exact-stack regional mechanics evidence exist. It also supplies the
source-geometry prerequisite for the blood spatial owner without promoting the
synthetic ABI38 blood receipt to anatomical completion.

The follow-on [source-frame moments receipt](ORGAN_GEOMETRY_MOMENTS_20260913.md)
computes algebraic volume and first/second spatial moments for the 357 members
that are single closed components. It leaves the inventory itself immutable and
does not change the source-only admission boundary.
