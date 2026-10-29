# Connected coarse tissue repair inventory

The physical skin/fat volume still overlaps bones. This change extracts deterministic boundary-face and parent-cell witnesses directly from each material-solid Boolean intersection, so the measured regions can be repaired together. It changes no geometry, force, contact rule, mass, or runtime admission.

The complete 185-bank reference-pose pass takes 0.49 seconds on SSH Numi. Its 185 positive material/bank volumes agree bit for bit with retained inventory 429. Surviving material-boundary provenance identifies 1,486 cells and 1,198 nodes; complete vertex stars merge into five disjoint repair candidates of 4,375, 708, 523, 914 and 1,639 cells. No positive pair lacks a boundary witness in this run. Overlapping per-bank volumes are not added as a physical union.

The focused check verifies ancestry after Boolean face splitting, distinguishes fully contained solids without a material-boundary witness, and merges overlapping node stars before patch edits. Patch closure, anatomical fitting, cell quality, conservative mass transfer and native motion are separate pending gates. This inventory does not prove those gates.

Geometry uses actual canonical Float32 physical vertices promoted to Float64 for offline queries, and the unchanged hash-pinned bone operands. It is a reference assembly, not a measured individual.
