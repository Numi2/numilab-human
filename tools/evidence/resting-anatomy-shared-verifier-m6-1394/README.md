Run on SSH Numi:

    PYTHONPATH=src python -m numilab_human.resting_anatomy_verify --pack CAPTURE.mrvpack --capture-receipt CAPTURE.receipt.json --changed 51011:471 --output audit.json

All anatomical layers are targets by default. Typed semantic/stable IDs prevent aliasing; duplicate primitive chunks are retained. Exact packed-coordinate identification affects predicates only and reports original and quotient topology. Containment uses rational metres and requires a closed embedded target. Open boundaries remain open. Declared interface face pairs do not prove connected vessel lumens.

The Python API compares emitted Float32 positions with the existing rigid/material-parent owner reconstruction. It does not introduce another transform or simulator. Passing selected checks does not admit whole anatomy, mechanics, physiology or clinical validity. The current N27 capture decoded 873 typed anatomical surfaces; only receipt/section/pose input checks were run for that capture in this increment.

The same entrypoint now accepts the existing accepted-capture index. Its pinned source inputs and capture receipts are checked by `audit_capture_sequence`; checks stop at the first failed capture. Require closure explicitly for a changed solid:

    PYTHONPATH=src python -m numilab_human.resting_anatomy_verify --capture-manifest accepted-capture-index.json --manifest-sha256 MANIFEST_SHA --source-pin /absolute/source.nhtissue=SOURCE_SHA --changed 51005:132 --required-closed 51005:132 --output motion-audit.json

Repeat `--source-pin`, `--changed` and `--required-closed` as needed. No pose reconstruction, candidate replacement or interface waiver is implicit. Source-bound reciprocal/material interface declarations still use the same Python API; a CLI-only sequence with no declarations cannot admit those contacts. The CLI applies any explicit intended enclosure to every indexed capture and never exempts boundary crossings.

Verification of this CLI increment on SSH Numi: 31 tests and 6 subtests passed, including changed-manifest rejection before capture loading, sequence dispatch and incompatible-input rejection; Ruff and diff checks passed. These are software checks, not new anatomical or native-run qualification.

`passive_tissue_parent_carriers.validate_compiled_core_weights` provides the shared preparation check for keyed core fields. It validates original Float64 source rows with the unchanged 1e-12 partition guard, then requires active native weights (Float32 or exactly promoted doubles) to equal the keyed source Float32 casts. It checks ordered body identities and unique source/active GIDs; it neither normalizes nor alters weights. Compiled row-sum residuals are diagnostic, not source-partition failures. Focused carrier/verifier checks: 40 tests and 12 subtests passed on SSH Numi; Ruff and diff checks passed.
