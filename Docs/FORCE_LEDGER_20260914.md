# Per-DoF force audit — 2026-09-14

The generalized-force ledger now emits all 128 coordinate rows, rather than
only the ranked worst rows. Each row retains the authoritative net force,
force scale, normalized residual, acceleration when supplied, and every
source contribution with its owner. The required sources are gravity,
muscle/tendon, joint equality, joint limit, support contact, and passive
tissue.

This is the diagnostic needed before a dynamic standing step: it can show
which articulated coordinate remains unbalanced even when the six floating
root equations close. `per_dof_source_audit` describes source coverage; it
does not turn a large residual into a pass. Force convergence, loaded
equilibrium, anatomical contact, activation calibration, and sustained
standing remain separate gates until a native full-body snapshot passes the
configured closure limits.

The existing command remains:

```sh
PYTHONPATH=src python3 -m numilab_human.force_ledger \
  --input generalized-force-snapshot.json \
  --output generalized-force-ledger.json
```
