# Tissue mass composition candidate — 2026-09-14

The new `tissue-mass-candidate` owner compiles the pinned organ geometry moments into a conservative composition record. It covers all 18 declared source regions and all 378 unique source members. It admits an explicit `1060 kg/m³` engineering density only for single closed organ-parenchyma components whose source moments are already computed. It emits deterministic zeroth, first, and central second candidate mass moments while retaining the source hash, region, class, and ownership boundary for every row.

The first receipt admits 342 organ-surface candidates with a total candidate mass of `2.7775569776377265 kg`. Thirty-six rows remain unresolved: 14 have disconnected or defective source topology, eight are shared seam members declared by more than one region, and the remaining rows are vessel or otherwise non-parenchymal classes. Shared members are never counted twice; their `region_ids` remain visible and their mass admission is `shared_source_member_overlap_unresolved`.

The profile keeps the actual Human ownership boundaries explicit:

- blood remains owned by the CVSim21 aggregate owner candidate; no tissue exchange or anatomical lumen is assigned;
- vessel surfaces remain surface integrals, not wall or lumen volumes;
- skeletal-muscle tissue partition remains separate from the MyoSim rigid-body mass owner;
- fat, skin, tendon, and fascia have no admitted physical volume or calibrated material;
- no candidate row becomes a mechanical mass owner.

The record is therefore a useful mass-composition input for the next disjoint-volume and calibration step. It does not qualify organ mechanics, blood/tissue transfer, materials, standing, walking, or subject calibration. The immutable receipt is [`receipt-v2.json`](media/tissue-mass-candidate-20260914/receipt-v2.json), SHA-256 `46bfeedaca172805c38c8f444e1c6061cde0af6f01a62369baf69c5f1d82f53c`.

Reproduce it with:

```sh
PYTHONPATH=src python3 -m numilab_human.tissue_mass_candidate \
  --output Docs/media/tissue-mass-candidate-20260914/receipt-v2.json
```

The execution registry now records this as `organs.mass_composition`, between source inventory and organ mechanics. The refreshed [gap execution report](media/gap-execution-20260914/report-tissue-mass-candidate-current-20260914.json) remains `integrated_qualification=not_assessed`; adding the candidate does not promote the unresolved physical gates.
