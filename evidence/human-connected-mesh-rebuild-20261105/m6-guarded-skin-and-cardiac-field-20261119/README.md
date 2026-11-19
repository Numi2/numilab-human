# Shared cardiac field correction and guarded skin enclosure

This is a diagnostic and authoring checkpoint for the existing Numi Human
resting scene on SSH Numi's Apple M6. The coupled variable-pleural-pressure
run is still blocked by the cardiac geometry gate. The previous failed native
trajectory reached 5.558 s. The original model remains the default. Every native run here uses
the existing pressure, contact and acceptance operators and a 2 ms
physical timestep.

## Cardiac field

The retained source at zero chamber coordinates cleared the full seven-surface
check. The prior field with the tight local retriangulation had 18, 24 and 28
crossings at the original box boundary, first rejected candidate and largest
sampled right atrium excursion. A shared field correction modifies 38 welded
source rows on two chamber basis vectors. It leaves the original source
positions and surface ownership intact and clears all three fitted poses.
The Apple M6 probe reproduces the checked fused Float32 positions bit for bit.
All seven oriented surfaces remain closed. The recalculated signed-volume
polynomial delta gives a uniform relative-volume bound of 1.816e-7 over
the declared coordinate cube.

The same field has exact crossings at 16 of the 17 additional calibration
poses. The right atrium still crosses the myocardial bank at 15 of those
poses. These are authoring probes on an existing calibration trajectory.
They do not provide a certified pressure interval or new accepted native
steps. The reported per-pose exact counts are lower bounds at the states
where the interval tool capped its witness output. Even a radius-1e-6
box around the deepest fitted pose is uncertified. A first fit to the
larger set reduced its linear constraints but inverted triangles on an
exact finite-step area check, so it was discarded. The nonlinear area-preserving retry retained positive
faces and still failed all 20 probes. The independent shared local affine
fits also left crossings. No tested atlas covers the full trajectory.

## Skin enclosure

The separating-plane fit was authored jointly at the existing 4 s
variable-pressure capture and 16 s constant-pressure capture. The first
candidate reduced the total and introduced new crossings. The guarded
subset retires the source vertices responsible for those new witnesses
and keeps the original triangles and bindings. Closed swept boxes cover
all per-vertex choices between preserved and proposed positions. The
exact collision predicates verified the baseline and candidate changed
faces against all relevant typed banks. Exact skin self-tests ran on the
surviving subset. The final fit pins all 32 source bed anchors and keeps
1,122 Float32 position changes. A first version omitted one anchor and
was rejected at preflight.

| Snapshot | Skin-to-tissue pairs before | After | New pairs | Skin self-pairs |
| --- | ---: | ---: | ---: | ---: |
| 4 s | 13,597 | 13,374 | 0 | 36 |
| 16 s | 14,606 | 14,498 | 0 | 55 |
| Initial pose | 8,596 | 8,586 | 0 | 0 |

These values describe the anchor-pinned candidate on frozen authoring
poses. Its 16 s native replay reproduced the reported late count and
kept the body, respiration, cardiac buffers and support impulse traces
byte-identical to the control. Four bed section rows changed, including
one projected area increase of about 9.25%, so the candidate failed
section parity. Source NHCNT identities are one-based. Section
quadrature also depends on incident faces, regional minima, and the
nearest registered thickness samples. The section dependency probe
identified these owners in the actual native release pose.

The revised candidate protects 9,306 source rows, with 1,072 Float32
position edits outside the protected footprint. The one-step native
preflight reproduced all 32 bed sections and matched the accepted body
and respiration digests. Exact audits of the three authoring poses
found no introduced pairs and retained the existing self-pair counts.

| Snapshot | Skin-to-tissue pairs before | After | New pairs | Skin self-pairs |
| --- | ---: | ---: | ---: | ---: |
| Initial pose | 8,596 | 8,594 | 0 | 0 |
| 4 s | 13,597 | 13,406 | 0 | 36 |
| 16 s | 14,606 | 14,506 | 0 | 55 |

The revised candidate completed all 8,000 native authored-pressure
steps in 160.717 wall seconds, with 3 breaths
and 19 complete filling and ejection cycles.
Measured throughput was 0.10780× during motion
and 0.09955× from launch to exit. All body, cardiac, respiratory, support impulse, momentum and bed section comparisons matched byte for byte.
The native terminal exact audit counted 14,506 skin-to-tissue pairs,
including 0 new pairs, and 55 skin self-pairs.
The terminal bed supported 705.65 N and the measured root assistance remained zero.

Continuous full-body clearance remains open.

## Native controls

All observer batching controls completed 8,000 accepted 2 ms steps and kept
the terminal body, respiration, three cardiac buffers, transaction and
geometry byte-identical. Each retained the same bed section data, with zero
sampled root assistance. The observer emitted fewer rows at larger cap
sizes and retained the same presentation cadence.

| Authored-pressure control | Wall seconds | Launch-to-exit rate | Sustained rate |
| --- | ---: | ---: | ---: |
| Prior cap 8 | 162.045 | 0.09874× | 0.10689× |
| Cap 16 | 160.981 | 0.09939× | 0.10758× |
| Cap 32 | 160.677 | 0.09958× | 0.10782× |
| Cap 32, verified initial-pack clone | 161.246 | 0.09923× | 0.10769× |

Every launch-to-exit control remains below 0.1×. The initial-pack cache did
not improve the matched larger batch and is not a default change.

## Reproduction

Use the neighboring native checkpoints and the original local, source-licensed
assets on SSH Numi. The parent checkpoint is `34d6be86e`
and includes the exact geometry predicate source lineage and previous
semantic bank inventory. This folder contains the new authoring tools,
matched launch receipts, witness tables and hashes. Large simulation
and authoring geometry stays on Numi.

From the `resting-clearance-and-atrium-004` checkout, use the retained
Python environment at
`/Users/numi/numi-human-aortic-interface-20261015/.venv/bin/python` and
include `../resting-surface-clearance-001/src` on `PYTHONPATH`.
The geometry helper inputs are in the adjacent `002` and `003`
scratch checkouts. The pressure path and native executable are pinned in
`runs/native-segment*-launch.json`.

To reproduce the support check, run `build_bed_footprint.py` followed
by the `baseline` and `anchor-pinned` cases of `run_bed_footprint.py`.
Run `pin_bed_footprint.py` and set `NUMI_AUTHOR_BED_FOOTPRINT=1` for
`trim_three_poses.py`, `author_guarded_skin.py` and
`build_guarded_skin.py`. The `footprint-pinned` preflight must pass
before the `full` replay. Run the retained exact MRVPACK2 audit and
`verify_guarded_native.py --footprint` against the resulting capture.
