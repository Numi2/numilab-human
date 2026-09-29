# Reproduce the complete compiled Human surface census

The two deterministic `.tar.gz` archives contain each run's `identity.json`,
`summary.json`, and all 579 per-surface `rows/NNN.json` files. The gate reads
these archives directly, checks every row hash, rehashes every compiled
source-mesh slice, validates the 19 repair candidates, and reruns the five
retained native packet/profile checks. `selected-surface-gate.json` is its
terminal report. The original indexed and exact-coordinate quotient results
must be kept distinct.

From the `numilab-human` checkout, use `PYTHONPATH=src` and the pinned
`.venv-mujoco312/bin/python` to run
`python -m numilab_human.whole_body_surface_gate --help`. The exact invocation
and input paths are in `gate-command.json`; substitute these public census
archives for its two `Build` archive paths. The `indexed-command.json` and
`quotient-command.json` commands regenerate all rows from the unchanged 579-
surface payload when the pinned source archives and manifests are available.

The result is a per-surface geometry gate only. It does not admit clinical
anatomy, cross-surface containment, material volume or mechanics.
