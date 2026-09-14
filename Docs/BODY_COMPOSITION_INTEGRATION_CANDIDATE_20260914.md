# Body composition integration candidate

The `HumanPack.body-composition-integration-candidate.v1` compiler joins the
current one-adult-male source package at the boundary between organ identity,
blood transport, oxygen exchange, cardiac blood budgeting, muscle activation,
and surface identity. It is available through:

```sh
numi human body-composition-integration --output /private/tmp/body-composition-integration.json
```

The candidate is source-bound and hash records the exact upstream files. It
checks that the current graph contains:

- 378 unique organ members across the 18-region mass inventory, with 342
  single-closed candidates and eight shared members retained as unresolved;
- 329 unique members assigned to the seven exact-clock blood-transport beds,
  with 511 accepted and one rejected `12,500 ns` step and neutral rollback;
- 150 distinct muscle/tendon surface identities covering the current surface
  rows and all 416 activation route identities, while retaining 238 routes
  without a surface binding;
- the same seven beds and clock in the bidirectional oxygen-exchange candidate;
- both cardiac geometry conventions bound to the common `0.39974235600733804
  kg` hydraulic blood-mass budget without selecting a physical owner.

The resulting receipt is a graph and ownership certificate. It does not add
candidate masses to rigid dynamics, create anatomical blood or lumen volume,
assign skeletal-muscle, fat, skin, tendon, or organ mechanical volume, or
claim material or subject calibration. Those are the next physical owners and
remain open. The source receipt hashes, counts, and owner-null assertions make
this join fail closed if an upstream identity, route, bed, clock, or ownership
boundary changes. It also emits stable digests for the organ, blood-member,
and muscle/tendon surface identity sets, along with the subset/disjointness
checks used to construct the join.

The candidate therefore advances the organ, systemic-physiology, and muscle
rows from disconnected source evidence to one reproducible integration
boundary while preserving the completion gates for calibrated mechanics,
anatomical blood/tissue transfer, standing, recovery, and walking.
