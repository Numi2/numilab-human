# Patellofemoral reference path comparison — 2026-10-03

This exploratory, source-bound comparison measures the pinned MyoSim left
patella path against the archived Open Knee oks003 specimen at selected flexion
angles through the archive's 89.9445° endpoint. It adds a reproducible
cross-specimen diagnostic; it does not select a corrected MyoSim joint law.

The owner command builds the pinned MyoSim source, projects its equalities, and
retains only source-range-valid samples from a 0.5° sweep. It reads the 140
accepted rigid observations retained from the Open Knee FEBio 2.9.1 archive.
Patella rotation is compared as the change in shortest relative-quaternion
angle from each model's own starting state. Translation is the change in rigid
origin, expressed as a magnitude from that model's starting state; MyoSim uses
the patella body origin and Open Knee reports the patella rigid-body center of
mass. Those translation observables are not the same anatomical point.

| Knee flexion | MyoSim patella-origin displacement | Open Knee patella-COM displacement | MyoSim orientation change | Open Knee orientation change |
| ---: | ---: | ---: | ---: | ---: |
| 30° | 26.315 mm | 28.658 mm | 14.640° | 16.179° |
| 60° | 49.051 mm | 47.939 mm | 43.627° | 37.368° |
| 89.945° (archived endpoint) | 65.771 mm | 64.955 mm | 72.999° | 58.522° |

The origin-displacement proxies are close at the archived endpoint, while the
MyoSim orientation excursion is larger by 14.477°. This is useful for narrowing
the diagnostic question, but not for calibration: the models use different
specimens and mechanics, there is no anatomical frame registration between
them, the Open Knee input is a passive quasi-static run, and its original solver
binary is not identified or reproduced locally.

The comparison also confirms that the declared MyoSim source joint ranges are
not valid across the full sampled path: 26 of 181 sampled poses fall outside at
least one source range, at sampled angles 0.5–1.0°, 42.5–50.0°, and
60.5–64.0°. The three reported comparison poses themselves pass the source-range
checks. Only their per-pose kinematic values are compared; the valid portions
form separate intervals, so no continuous 0–90° path or collision-free
trajectory is claimed. The existing exact mesh audit still reports source
patella/femur intersections.

Reproduce with the pinned MyoSim Python environment and a new output path:

```sh
PYTHONPATH=Sources/myosim/checkout numi human \
  myosim-patellofemoral-reference-path-audit \
  --sources Sources \
  --observations Docs/media/open-knee-reference-20261001/archived-observations.json.gz \
  --archive-audit Docs/media/open-knee-reference-20261001/recovered-source-audit.json \
  --intersection-receipt Docs/media/numi-human-patellofemoral-pose-intersections-20261003/receipt.json \
  --output Build/patellofemoral-reference-path-replay.json \
  --python /private/tmp/numi-human-myosim-20261003/bin/python
```

The immutable [receipt](media/numi-human-patellofemoral-reference-path-20261003/receipt.json)
binds the MyoSim source overlays, Open Knee observations and archive audit, and
the earlier exact-intersection failure. It records 185 negative-Jacobian trial
diagnostics in the reference log. Clinical anatomy, loaded tracking, continuous
motion, contact, source-path correction and whole-Human qualification remain
false. The next repair needs source-bound anatomical frame registration and a
loaded patellar path that preserves quadriceps/patellar-tendon ownership while
passing the existing range and surface-intersection gates.

The focused analysis tests pass (5 passed); Ruff and `git diff --check` pass.
