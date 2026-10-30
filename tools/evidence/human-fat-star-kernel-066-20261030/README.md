# Human fat partial construction evidence - 2026-10-30

## Scope

This package records a bounded, zero-coordinate construction repair for the current fat-volume preparation. It includes the earlier 061-064 diagnostics needed to understand the input and the corrected, guarded 065 contraction. The raw 064 mesh is the immediate input to both 065 executions.

The original 065 script had a fail-open issue: it computed and recorded the simplicial link condition, but continued with the contraction when that condition was false. The guarded 065 derivative rejects before the first cell mutation when the link condition fails. It also checks preservation, zero-cell removal, duplicate cells, tet-face incidence, boundary-edge incidence and the source-cell partition before writing a candidate.

## Results

- The 061 exact-binary-rational orientation audit examined 853,727 fat cells: 16 were exact F32 zero and 93 exact F32 negative. Its standard float64-on-F32 signs agreed with the exact signs for the 109 nonpositive rows.
- 062 produced six local star-kernel constructions, and its separate exact patch verifier passed those six local patches. It still reported 16 zero and 86 negative cells in its whole candidate.
- 063 and 064 are separate fixed-boundary repair attempts from the fat.1 source. 064 emitted the input used here: 853,716 cells, with 16 F32-zero and 70 F32-negative cells.
- 065 contracted four exactly coincident construction vertices onto a retained node, removed 16 repeated-node, exact-zero cells, and retained all 245,996 original source points bitwise and in the cell mesh. It produced 853,700 cells and 246,506 points. The result has zero F32-zero cells, 70 F32-negative cells, no duplicate cells, no nonmanifold tet faces, and no boundary edge with incidence other than two.
- The guarded replay reproduced every NPZ key and array value from the frozen 065 candidate. The candidate archive SHA-256 is 480c3137e733ddcc4646589d23ad67a3ade01b86856d068f0814aa10d3144c1d for both files.

## Controls and fail-closed fix

Four focused controls passed: a stellar 1-to-4 split whose construction point coincides with a source corner; rejection of a one-subnormal F32 near-coincidence; rejection of merging two original source rows; and rejection of a nontrivial failed-link contraction before input cells are mutated. The last fixture demonstrates that the frozen unguarded 065 logic could create a duplicate nonzero cell after a failed link.

## Limits

This is a partial construction result, not a physically qualified volume. Seventy negative F32 cells remain. The recorded boundary-subdivision inventory still has 196 missing original PLC facets and 400 added boundary facets. Source boundary restoration, all-positive F32 cells, original material/mass mapping, native import, and accepted-state simulation remain unqualified. No runtime simulation, physical admission, or mass-conservation claim is made here.

The included storage aggregate is independent private-text lossless-compression verification evidence; it does not change the geometry status.
