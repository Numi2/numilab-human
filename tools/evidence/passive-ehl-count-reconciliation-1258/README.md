# Passive EHL serialized triangle-count reconciliation

This is a metadata-only correction in the existing NHTISS composition owner. Stable ID 23's direct-parent manifest reported 1,480 triangles, while its serialized ABI5 row contains 4,434 indices (1,478 faces). The row's vertex records, bindings, weights, and local faces are byte-identical between the direct parent and the new candidate.

The child manifest now reconciles the stable ID 23 triangle count to 1,478 and records the inherited mismatch explicitly. It retains source_component_selection.retained_triangle_count=1480 and source_precision_repair.removed_triangle_count=2 unchanged as historical provenance. The NHTISS payload is byte-identical to the prior FHL compose-004 payload; the receipt changes only to bind the new manifest and paths.

The focused composer suite passed 29 tests, including a regression that starts from a deliberately stale inherited stable-23 count and exercises FHL composition plus receipt binding. This is not a geometry change or native/anatomical qualification.

Exact paths, hashes, command, and comparison details are in composition-recheck.json and composition-command.txt. No binary payload was copied into this evidence folder.
