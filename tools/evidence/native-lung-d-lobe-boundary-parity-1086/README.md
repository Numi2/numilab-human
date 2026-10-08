# Native D311-lobe boundary audit adapter (1085/1086)

This bundle preserves the immutable 1084 native feature index and adds a narrow adapter for D311/lobe events that 1084 leaves unclassified. It aligns the native audit with the already-published exact source rule in Human revision 46b3c0b. It changes audit labeling only: it does not alter geometry, the exact intersection predicates, the retained native scan, or any acceptance gate, and it does not qualify the 1078 anatomy.

## What the correction does

- Uses vertices from the accepted MRV pack in the actual accepted-pose world Float32 coordinate frame, represented as exact binary32-lattice integer keys.
- Normalizes either input argument order so the source helper receives the D311 face and the lobe face in their declared roles. The step-0 ledger stores lobe then D311.
- Requires the validated D-to-lobe map and exact equality of the D and lobe full-union boundary sets. It applies the source helper's exact reciprocal-face/adjacency rules or requires every exact witness to lie on one concrete declared union-boundary edge.
- Never rounds a noninteger Fraction witness. Map disagreement and unsupported witnesses fail closed. All non-D/lobe classifications are delegated to 1084.

The original corrected step-0 scan recorded 13,050 D/lobe witnesses as unclassified. Replaying those retained events through the repository adapter labels 4,754 for lobe 305, 4,978 for 306, 2,662 for 307, and 656 for 308 as exact declared union-boundary contacts; lobe 309 has zero such events. Each accepted witness is one exact boundary vertex. There are zero residual events rejected by the published source helper in this retained event set. The original ledger labels remain unchanged; this replay is not a new scan and does not prove physical clearance.

The focused tests cover exact boundary vertices, off-boundary rejection, witnesses on different edges, noninteger witness rejection without rounding, reversed D/lobe argument order, native-map mismatch, and delegation of non-D/lobe behavior. The retained test log reports 14 passing tests.

As separate context, the pinned 1084 report records all 1,913 exhaustive sampled native-frame feature lookups matching, within a 151,126-event step-0 census, and a sampled lookup speedup of 135.73x. That is a scoped measurement from the earlier 1084 indexing check, not a general performance claim.

## Reproduce on the pinned Mac mini

The runner reads retained external native inputs at the absolute paths recorded in `retained-artifacts.json`; it refuses to overwrite an output directory. Run it from this bundle, choosing a new disposable destination:

    cd /Users/n/numi-human-d-lobe-adapter-publication-001/tools/evidence/native-lung-d-lobe-boundary-parity-1086
    /Users/n/numi-human-prep-venv-20261005/bin/python replay_retained_step0_events.py --out /Users/n/numi-human-resting-evidence-20261005/native-lung-d-lobe-boundary-parity-1086/repository-adapter-replay-reproduction-003

The event-output SHA-256 should match the retained replay event file recorded in the replay report. To run the focused test suite:

    /Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q test_native_feature_index_1085.py

Full input and output hashes are recorded in `retained-artifacts.json`; the local evidence files are covered by `SHA256SUMS.txt`.
