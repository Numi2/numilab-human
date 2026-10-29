# Connected tissue fit: construction and initialization-domain checks

The preparation driver now fails on a changed canonical midpoint baseline, validates its execution budget, accepts a unique output directory, saves optimized geometry before metadata serialization, and fixes a report-only quantile error. The existing candidate and executed producing script remain unchanged.

The full zero-iteration preparation/serialization control passed on SSH Numi in 3.25 seconds. Its 7,491 emitted Float32 nodes and 29,642 cells reproduce the pinned conforming control exactly; the checkpoint matches the final candidate, all cells are positive, and invalid iteration input is rejected before output creation. This is a preparation regression check.

The separately retained run-006 connected fit passes local cell, material-density, thickness and pin checks. Its first bone-occupancy gate fails: body 23 / bone 29 intersects 6.0823 mL of skin and 70.8807 mL of fat. These are material intersections for this single bank, not a union across banks. It remains an unadmitted construction candidate. Inserted-midpoint correction alone leaves original penetrating boundary rows; further connected boundary fitting is required.

The initialization interpretation addendum distinguishes historical NPZ coordinates from the actual native import contract. It corrects an earlier diagnostic interpretation without discarding its raw evidence. Immutable 330 mass remains separate from operational source geometry.

No Runtime, GPU step, accepted motion, native anatomical qualification or performance result is claimed. The native scene and accepted fine geometry are not changed by this increment.
