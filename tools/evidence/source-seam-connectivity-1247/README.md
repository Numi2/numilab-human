# Preserve exact OBJ seam connectivity during component selection

The largest-component selector joined faces by OBJ vertex index. BodyParts3D
uses separate vertex records along shared coordinate edges, so this selector
discarded valid source faces and opened otherwise closed muscle surfaces.

The existing selector now discovers adjacency through exact source-coordinate
edges, retaining original selected vertex records, face order and winding.
It performs no coordinate rounding, tolerance weld, vertex movement, inferred
cap, or triangulation. Point-only contacts remain separate.

The source-bound comparison covers the exact six affected source members:

| Stable ID | Member | Restored source faces | Boundary edges before / after |
| --- | --- | ---: | ---: |
| 7 / 8 | FJ1405 / FJ1405M | 617 each | 543 / 0 each |
| 23 | FJ1408 | 184 | 186 / 0 |
| 24 | FJ1408M | 183 | 187 / 0 |
| 27 / 28 | FJ1415 / FJ1415M | 81 each | 85 / 0 each |

All six selected source sheets have zero boundary, nonmanifold, orientation
mismatch and degenerate face counts under exact coordinate topology checks.
The exact-area supplement also finds no collinear source triangles, using
integer arithmetic on the dyadic lattice of the parsed source coordinates.
Every previously retained oriented source face remains. These are source mesh
checks, not registered anatomy or native physics qualification. Source
provenance remains BodyParts3D 4.0 (CC BY 4.0); no source meshes are redistributed
in this evidence bundle. The current runtime's inferred reconstructed tendon
rows 7/8 have not been replaced by these source outputs.

Validation on the SSH Mac mini:

- Three focused component-selection tests pass, including the new seam and
  nearby-coordinate regressions. The existing shard test shares only one
  point with the dominant sheet and covers the point-contact boundary.
- The importer and opposite-face suites report 109 passes and eight failures
  in the candidate; the unmodified parent reports 107 passes and the same
  eight failures. The worktrees lack external Sources/Build fixtures. Full
  suite success is not claimed; exact logs and failure IDs are retained.
- source-selection-comparison.json pins the archive, six member identities,
  parent/candidate owners, predicate helper and runner. Inputs remain unchanged.
- Comparison attempt 001 failed before geometric comparisons because its
  isolated parent-function namespace lacked typing.Any. The corrected sibling
  is the completed attempt; both and the failure note are retained.

No native run or composed anatomy change is established by this commit.
Regenerated registered rows still require exact geometry and accepted-pose
checks before they can replace the current package.
