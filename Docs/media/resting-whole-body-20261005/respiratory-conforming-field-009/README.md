# Native respiratory geometry regression, 2026-10-05

The existing GPU respiratory coordinates now have an asset-preparation path
that conforms the lung and diaphragm triangles to their shared piecewise-affine
basal field. The physical owner remains NumiLab's Metal MyoSim/tendon and
thoracic update. This adds no tissue elements or time integrator.

All preparation, tests, compilation and native execution for this increment ran
on `ssh macmini` (Apple M4 Pro). The native runtime is the frozen
`/Users/n/numi-human-resting-conforming-{source,build}-009`; its base revision,
exact source diff, executable/Metal hashes, arguments and environment are
retained here. Large source assets, accepted MRVPACKs and the continuous movie
remain under `/Users/n/numi-human-resting-evidence-20261005/`.

The `thorax-conforming-field-009` preparation contracts only edges shorter than
0.125 micrometres. A shared coordinate quotient preserves reciprocal faces;
each connected group selects an existing minimax representative. Fifty unique
points move, by at most 0.124895 micrometres. All five lobes and the diaphragm
remain closed and consistently oriented. The previous unassigned diaphragm
boundary was two paths separated by 1.86 nanometres; its explicitly validated
closure repairs a numerical seam, not an identified anatomical hiatus.

The inferred passive pleural proxy is the two-component external lung union.
It retains source provenance and excludes the explicitly identified tiny
source fragment. It does not represent parietal pleura, fissure lining,
pleural-fluid mechanics or measured individual anatomy.

The combined scene (`cardiac-wall-binding-005`) completed 3,000 native steps:
6.000000285 simulated seconds in 65.570607125 native wall seconds (0.0915044x
real time; wrapper 69.467797 seconds). The coupled rejection/retry probe passed.
All displayed geometry-status checks passed. This runtime includes the known
unresolved cardiac-wall mapping; numerical success is not whole-body anatomy
admission.

The exact accepted-geometry regression checks all 31 previously retained
lung/diaphragm parent-pair witnesses plus one separate patch-boundary witness.
At steps 0, 639, 2783 and 2999 there are zero forbidden intersections and zero
degenerate triangles on surfaces 305–311. The earlier one-micrometre quantized
candidate had 3, 3, 6 and 3 forbidden pairs respectively; its negative report is
retained. This is a targeted regression, not an exhaustive thorax-clearance
certificate or five-minute qualification.

Six field tests and five pleura tests pass. Generate the field through
`python -m numilab_human.resting_respiratory_conforming_field` with the existing
`--input`, `--receipt`, `--config`, `--output` arguments and
`--short-edge-resolution-m 1.25e-7`. The exact input identities and resulting
configuration are recorded in `source-result.json` and the Mini receipt.
Generate the passive envelope with
`python -m numilab_human.resting_pleura_proxy --base-payload ... --base-receipt ... --output-dir ...`.

Remaining work includes full-cycle thorax/rib clearance, hepatic registration,
cardiac-wall embeddedness, physiological endurance/intervention qualification,
and performance. Neither these tests nor the six-second run establishes
physiological plausibility or clinical validity.
