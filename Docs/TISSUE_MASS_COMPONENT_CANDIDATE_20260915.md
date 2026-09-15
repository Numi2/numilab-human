# Component-sum tissue mass candidate — 2026-09-15

The [immutable candidate receipt](media/tissue-mass-candidate-20260915/receipt-v1.json)
consumes the conservative disconnected-organ moment receipt and applies the
existing explicit `1060 kg/m³` engineering density only as a source-bound
candidate.  It covers all 18 regions and 378 source members, increasing the
candidate rows from 342 to 349 by admitting the seven disconnected members
whose individually closed components have pairwise disjoint source bounds.

The eight shared source members remain rejected from double counting.  The
seven overlapping-bound multi-component members and seven topology-defective
members remain unresolved.  The candidate emits deterministic zeroth, first,
and central second mass moments, but it does not establish disjoint biological
volumes, lumen or wall ownership, physical organ mass, blood transfer,
materials, subject calibration, or mechanics.  Fat and skeletal-muscle tissue
remain separate unresolved ownership domains.

The receipt was generated with:

```sh
PYTHONPATH=src python -m numilab_human.tissue_mass_candidate \
  --moments Docs/media/organ-geometry-component-moments-20260915/receipt-v1.json \
  --output Docs/media/tissue-mass-candidate-20260915/receipt-v1.json
```

SHA-256: `06374e58b5df14b4160c7c8616b8352b2d44967aea885f5523a536786329f93e`.
