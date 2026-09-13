# Source-frame organ geometry moments, 13 September 2026

The pinned BodyParts3D inventory now has a separate source-moment receipt at
[media/organ-geometry-moments-20260913/moments.json](media/organ-geometry-moments-20260913/moments.json).
It recomputes the exact source archive and inventory, then integrates oriented
surface tetrahedra only for single closed components.

The result covers all 378 unique members and 386 declared memberships. It
computes positive source-frame volumes and first/second spatial moments for
357 members. Fourteen members are closed candidates with multiple disconnected
face components and seven retain the previously recorded source topology
defects. Those 21 members remain without moments. The receipt preserves every
status and source hash.

Run the owner with:

```sh
numi human-organ-geometry-moments \
  --output Docs/media/organ-geometry-moments-20260913/moments.json
python3 tools/verify_organ_geometry_moments_20260913.py \
  Docs/media/organ-geometry-moments-20260913/moments.json
```

The integrals are algebraic source-surface quantities. They do not become
physical organ volumes, hydraulic CVSim volumes, density or mass owners. No
union, self-intersection or interdomain-overlap admission, body-frame
registration, material assignment, blood-mass assignment, or physical stepping
is performed. The receipt is therefore a source-data prerequisite for blood
and organ mechanics, not anatomical completion.
