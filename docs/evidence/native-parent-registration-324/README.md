# Native parent registration and exact near-tie handling

This change registers repaired source geometry without relaxing the 20 mm parent-registration bound. Previously rejected near ties now receive an exact rational nearest-simplex certificate over every same-material parent tetrahedron, pruned only by exact AABB lower bounds. Distinct equal-distance projections still reject. The ordinary query path retains bitwise comparison parity.

Both surface and combined-volume-node producers use captured native Float32 matrix entries and their true inverse. Legacy quaternion/transpose behavior remains available when those matrices are absent. The donor-frame CSV follows the native importer column order.

Four focused test programs pass: exact simplex and ambiguity cases; source ancestry and full-86 weights; query bounds; combined volume-node registration with native/legacy frames and negative cases. The combined test was initially invoked without its required output-directory argument; the failed command and successful corrected command are retained.

Full O286 source registration completed in 79.64 seconds: 40,144 nodes, 80,243 faces, and 405 exact fallback cases, each with a unique exact nearest projection. Maximum parent residual was 15.679 mm. Fresh parent payloads matched the prior capture. Its registration queries ran before a CSV header-order correction; the evidence records the executed exporter hash and corrected exporter hash separately.

This is source preparation, not repaired physical-volume admission or motion qualification. It does not assert that the new geometry has been installed in the scene. Gzipped evidence preserves original bytes; manifest.json records raw and compressed hashes.
