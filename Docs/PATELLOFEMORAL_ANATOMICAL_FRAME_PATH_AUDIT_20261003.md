# Patellofemoral path in a registered femoral frame — 2026-10-03

This follow-up puts the pinned MyoSim left knee and the archived Open Knee
oks003 specimen in the same source-defined femoral anatomical frame. Open Knee
`Xf/Yf/Zf` axes map to MyoSim flexion/anterior/proximal axes using the current
left payload registration: a proper rotation (determinant 1.0) and a uniform
0.984216 scale, checked against 72.267 mm and 71.126 mm femoral condylar widths.
The transform is pinned in the [registration manifest](media/numi-human-patellofemoral-anatomical-frame-path-20261003/open-knee-oks003-left.manifest.json).

Positions below are changes in the patella inertial COM relative to the femur
inertial COM, from each model's own baseline. Rotation is the signed
femur-relative rotation-vector change. Components are flexion-axis, anterior,
and proximal, in that order.

| Knee flexion | Open Knee COM change (mm) | MyoSim COM change (mm) | Open Knee rotation change (°) | MyoSim rotation change (°) |
| ---: | ---: | ---: | ---: | ---: |
| 30° | `[-0.99, -4.90, -27.76]` | `[0.00, -8.35, -22.83]` | `[-15.90, 0.57, 2.96]` | `[-14.64, 0.00, 0.00]` |
| 60° | `[-2.21, -22.80, -41.25]` | `[0.00, -20.10, -39.23]` | `[-36.85, -3.49, 5.12]` | `[-43.63, 0.00, 0.00]` |
| 89.945° (archived endpoint) | `[1.91, -43.08, -47.19]` | `[0.00, -36.00, -45.91]` | `[-58.02, -7.53, 1.52]` | `[-73.00, 0.00, 0.00]` |

The anterior COM projection remains positive at all 140 accepted Open Knee
reference states (minimum 9.283 mm) and all 155 MyoSim source-range-valid
samples (minimum 10.476 mm). At the archived endpoint, the projections are
9.283 mm and 10.506 mm. This addresses the directional “backside” concern for
the sampled states. It is a COM projection, not signed surface clearance; the
prior literal-qpos0 audit separately places every MyoSim patella vertex
anterior to its source knee-anchor plane by at least 33.589 mm.

The remaining measured path gap is rotational. At the archived endpoint,
MyoSim has 14.983° more flexion-axis rotation and lacks the reference's
7.528° anterior-axis and 1.525° proximal-axis components. Position changes are
closer at 60° than at 30° or the endpoint, but the two models still represent
different specimens and different mechanics. The Open Knee record is a passive
quasi-static archive; its original FEBio binary is unidentified and was not
reproduced locally.

This audit does not choose a corrected MyoSim path. Its source-equality sweep
still omits 26 of 181 samples because projected source joint coordinates leave
their declared ranges. The separate 0.05-rad sweep retains 34 of 43 samples;
the exact source femur/patella meshes intersect in 56 range-valid side-pose
states (1,236 triangle pairs), and the registered surfaces intersect in 41
states (1,631 pairs). These discrete failures, absent active quadriceps loading,
and unmatched specimens prevent the comparison from qualifying continuous or
loaded tracking, cartilage contact, or clinical anatomy.

Reproduce with the pinned MyoSim Python environment and a new output path:

```sh
PYTHONPATH=Sources/myosim/checkout numi human \
  myosim-patellofemoral-anatomical-frame-audit \
  --sources Sources \
  --observations Docs/media/open-knee-reference-20261001/archived-observations.json.gz \
  --archive-audit Docs/media/open-knee-reference-20261001/recovered-source-audit.json \
  --intersection-receipt Docs/media/numi-human-patellofemoral-pose-intersections-20261003/receipt.json \
  --open-knee-manifest Docs/media/numi-human-patellofemoral-anatomical-frame-path-20261003/open-knee-oks003-left.manifest.json \
  --output Build/patellofemoral-anatomical-frame-path-replay.json \
  --python .venv-mujoco312/bin/python
```

The immutable [receipt](media/numi-human-patellofemoral-anatomical-frame-path-20261003/receipt.json)
binds the MyoSim archive and overlays, archived Open Knee observations, source
audit, femoral registration, and exact-intersection failure. The evidence is
exploratory and selects no source-path correction.
