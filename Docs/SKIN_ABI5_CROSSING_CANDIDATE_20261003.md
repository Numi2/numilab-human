# Current ABI 5 skin crossing candidate

The exact current ABI 5 native packs expose a fold that the 512-pixel side
renders do not: the unique bilateral 1.2-radian whole-body skin pack contains
75 self-intersecting triangle pairs, while the source-rest pack contains zero.
The right and left focus captures at each pose have identical whole-body pack
hashes; the 75 count is not a per-knee count. The full exact face-pair witnesses
are retained in [crossing-attribution.json](media/skin-current-crossing-candidate-20261003/crossing-attribution.json).

I tested three geodesic weight variants against the same right-focused native
M4 capture, with both knee coordinates equality projected to 1.2 radians. The
candidate changes the ABI 5 weight field only; the registered positions,
normals, body-binding table and triangle indices remain byte-identical. Each
count is for the entire packed shell.

| Blend | Radius | Exact pairs | Baseline pairs resolved | New pairs | Maximum vertex shift |
|---:|---:|---:|---:|---:|---:|
| 5% | 12 cm | 69 | 42 | 36 | 2.11 mm |
| 5% | 6 cm | 69 | 42 | 36 | 2.11 mm |
| 10% | 6 cm | 67 | 65 | 57 | 4.22 mm |

These outcomes retain the defect and exchange many old pairs for new ones. I
rejected the weight adjustment as a repair and stopped tuning it; the rendered
candidate is visually indistinguishable at this camera scale. The complete
run identities, output hashes, commands and exact audit rows are in
[results.json](media/skin-current-crossing-candidate-20261003/results.json).
The native packs and candidate payloads are retained locally under the ignored
`Build/skin-current-crossing-candidate-20261003/` directory.

This result does not establish that the patella or skin is anatomically
correct. The separate patellofemoral pose audit confirms anterior patella
placement at source rest, but still records registered-surface overlaps in
the flexed pose and source-mesh crossings across the sampled range
([pose audit](PATELLOFEMORAL_POSE_INTERSECTION_AUDIT_20261003.md)). The skin
outer sheet also remains open, and neither this visual payload nor these
static poses provide skin mechanics, contact, subject calibration, or clinical
validation. A valid correction needs source-anatomy constraints and held-out
poses, not a reduction in one sampled intersection count alone.
