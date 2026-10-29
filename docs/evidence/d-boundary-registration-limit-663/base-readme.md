# Explicit D boundary retirement compiler increment

This increment adds a fail-closed D boundary retirement path to the existing o-boundary-retirement-433 exporter. Publication base: b672eb49b703e63d1da9c74e8b65491ae93dae00. The exporter blob at that base is ad63b10701c42b37b1aa6aa79e0dc0c9f7ac87cdea7622f65582f9d1a63af2b1. The shared checkout and index were not changed.

## Preserved current006 D surface

Retained facets are authenticated against the frozen run006 candidate and its existing 530 refinement and 534 source-reference ledger. The path checks child-tet rows, normalized node ancestry, source labels, outward orientation, and exact emitted-F32 facet coordinates. It retains all 4,949 current006 D facets through that existing ancestry and covers all 4,777 original 330 D faces. Unchanged current006 facets do not need a new local Boolean claim; changed facets require exact oriented-F32 geometric binding to the pinned Boolean output, original-face ancestry, and matching orientation from the actual outward child tet.

The exporter independently protects the 149 original head/attachment nodes. The default 0.4 mm source-registration bound remains unchanged; an optional explicit reference-registration limit is documented below. New or changed D-derived nodes must carry original 330 tet/barycentric ancestry and stay within that bound. Existing current006 derived nodes are exempt from the new-node bound only when their emitted-F32 coordinates and source tet/barycentric ancestry remain exactly unchanged. The frozen current006 baseline has 86 already-derived D rows; row 7423 has an existing source residual of 0.43485 mm. Any altered or reparented baseline row is checked as new.

## Validation

- D exporter suite: 13 tests pass, including pinned saved-CSG tag/geometry checks, full frozen006-to-330 ancestry, actual outward-orientation rejection, and end-to-end source-only exporter wiring.
- Existing O-retirement regression: 5 tests pass.
- Existing operational-reference map control passes with pinned run015/run020/run021 inputs; the legacy 42-column map remains byte-identical to run015.
- The four Python source/test files compile. Added lines have no trailing whitespace.

The integration fixture moves one unprotected D source node by at most 1 micrometre, preserves positive child tets, supplies a synthetic identity Boolean output for incident facets, and verifies exact F32 orientation/coordinate binding. This tests exporter wiring only; it is not an anatomical or physical replacement mesh.

## Geometry and admission limits

The saved cavity-minus-bank32-f64-tagged.npz (2,524 points / 5,044 faces) is a diagnostic CSG output. Its source tags and geometric support are checked against the pinned shell and native Float32 bank replay. It has no TetGen child node/cell output or spliceable current006 D child-facet map, and previous local audits retain unresolved interface pairs. The latest region23 run003 report also describes F64 skin/fat diagnostic surfaces without a D-role replacement or child-volume candidate. No candidate is promoted here: physical admission is false, native steps are zero, and accepted physics steps are zero.

Implementation attempts and the superseded row-7423 bound failure are preserved in attempt-history.json. See checks.json for exact hashes and commands. The D test source requires --evidence-root <physical-transfer-root> because the pinned geometry and mass inputs are external to the repository.

## Optional parent-registration reference limit

The exporter now accepts `--maximum-parent-registration-distance-m METERS`. The default remains exactly `0.0004` m (0.4 mm); values must be finite and positive. The selected value is passed to the existing original-330 tet/barycentric source-residual validator and recorded in each D-retirement receipt as `declared_maximum_parent_registration_distance_m`. This setting controls only source registration during reference reconstruction. It does not change physical geometry, label/material checks, mass mapping, or any admission threshold.

The focused source-only test uses a synthetic 0.5 mm residual to show that the unchanged default rejects it and an explicitly declared 2.5 mm registration limit accepts it; a 2.6 mm residual still fails. The original 0.4 mm failure evidence remains intact: current006 derived D row 7423 has a 0.43485 mm source residual and is not exempt if changed or reparented. The separate region23 run003 registration failure also remains preserved (1,839 of 11,818 tested fat points exceeded 0.4 mm, maximum 2.446110 mm); this option has not retroactively changed that run.

This is an unpushed follow-up to the published base `335d68793eb6fee415a477b96e2a95cdfcca96b3`.
