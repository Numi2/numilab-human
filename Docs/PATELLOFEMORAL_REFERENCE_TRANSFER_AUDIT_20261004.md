# Open Knee path transfer against MyoSim source surfaces — 2026-10-04

The preregistered cross-specimen transfer did not clear the pinned MyoSim
left-knee source-mesh intersection gate. It relocated the crossings: over 26
fixed common flexion samples, the current equality path had 529 intersecting
triangle pairs in 21 states, while the transferred archived Open Knee path had
530 pairs in 13 states. The transfer reduced the number of affected sampled
states by eight but increased the total exact pair count by one. It is not a
usable source-path correction.

The fixed transfer adds Open Knee's observed femur-relative patellar COM and
orientation changes to the MyoSim source-equality `qpos0` transform using the
retained proper femoral-frame rotation and scale. It checks the same source
range-valid samples as the pinned intersection receipt within the Open Knee
archive's accepted range. The transfer moves the MyoSim patella COM by up to
7.173 mm and its orientation by up to 16.554 degrees from the current source
path. It clears some states around 17–37 degrees but leaves or increases
crossings through the higher-flexion region; for example, the exact pair count
changes from 31 to 42 at 54.431 degrees and from 20 to 42 at 68.755 degrees.

This tests rigid source-mesh geometry only. The Open Knee record is passive,
cross-specimen, and not reproduced with its original solver here. The transfer
is not represented by MyoSim's three patellar generalized coordinates; tendon
and ligament attachments were not moved, and loads, contact pressure,
continuous motion, and clinical tracking were not tested. No source file,
compiled artifact, or runtime state changed. The transfer result is retained
as failed evidence; the source-path issue remains open pending a same-subject
measured path or an implemented and force-checked patellar/contact owner.

The immutable [preregistered plan](media/numi-human-patellofemoral-reference-transfer-20261004/plan.json)
binds the fixed sample selection and decision rule. The [owner audit receipt](media/numi-human-patellofemoral-reference-transfer-20261004/receipt.json)
retains every control and transfer state, exact pair count, displacement,
orientation delta, source identities, and predicate hashes. The two
[runner setup/implementation failures](media/numi-human-patellofemoral-reference-transfer-20261004/attempt-001-setup-failure.json)
and [attempt 2 failure record](media/numi-human-patellofemoral-reference-transfer-20261004/attempt-002-implementation-failure.json)
are preserved; neither produced a transferred-geometry result.

Reproduce with the pinned MyoSim environment:

```sh
PYTHONPATH=Sources/myosim/checkout numi human \
  myosim-patellofemoral-reference-transfer-audit \
  --sources Sources \
  --observations Docs/media/open-knee-reference-20261001/archived-observations.json.gz \
  --archive-audit Docs/media/open-knee-reference-20261001/recovered-source-audit.json \
  --intersection-receipt Docs/media/numi-human-patellofemoral-pose-intersections-20261003/receipt.json \
  --open-knee-manifest Docs/media/numi-human-patellofemoral-anatomical-frame-path-20261003/open-knee-oks003-left.manifest.json \
  --plan Docs/media/numi-human-patellofemoral-reference-transfer-20261004/plan.json \
  --output <new-receipt-path> \
  --python .venv-mujoco312/bin/python
```

The audit exits with status 2 when the sampled intersection gate fails; it
still writes the complete receipt before returning that status.
