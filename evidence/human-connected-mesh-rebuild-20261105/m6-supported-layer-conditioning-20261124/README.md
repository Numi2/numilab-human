# Skin and fat under load on SSH Numi

The current dense reference import contains 314,014 nodes and 1,457,316 tetrahedra. Its native import records 3.452571 kg skin, 10.487108 kg fat, zero inverted or below-shape cells, zero below-volume cells, zero nonfinite values, zero density outliers, and zero unassigned nodal masses. The maximum nodal mass error is 1.39e−17 kg. The full-contact native control still rejected its third 7.8125 µs step. The regional model with reduced bed mechanics has a separate 16-second result.

## Why the dense load fails

The last retained Newton iteration reports 0.0317014 nonvascular residual against a 1.25e−5 mechanical threshold. The vascular norm is 1.25e−8. All 46 active linear cycles stayed within their budgets. Material backtracking limited the correction to 1.19e−7 of its proposed size. The first failing cell was at J = 0.35 and its full trial was J = −0.03859. The three tile witnesses are actual registered skin cells.

| Registered row | Geometric donor / anatomical region | Reference minimum altitude | Longest edge / minimum altitude | Protected boundary |
| --- | --- | ---: | ---: | --- |
| 137663 | 46 / 42 | 25.0 nm | 7,311 | Two skin/fat interface facets |
| 612693 | 20 / 128 | 373.3 nm | 11,127 | Two external skin facets |
| 1439803 | 43 / 20 | 2.99 nm | 40,389 | Two skin/fat interface facets |

All three have four vertices on one registered layer wall and a stored reference-to-release volume ratio of 1.0. Across the whole mesh 77,040 tetrahedra have a reference altitude below 1 µm. Of these, 43,335 use four vertices on the same O, I, D or C wall. These categories describe geometric conditioning. The complete mesh is still subject to the existing material and motion tests.

## What was actually tested

The tests used the native reference coordinates, the loaded release, and a saved two-step deformation from a separate native trial using the exact same skin/fat files. Removing the external sliver would change the nodal force vector in its same-material local star by 0.345 N against an original 0.261 N. Although that cell weighs only 3.88e−10 kg and its mass can be redistributed conservatively in that star, its force response rules out deletion.

Across 1,635 fixed-interface local candidates, four passed the wall and reference-shape checks and none passed the downstream mechanical tests. The alternatives include local connectivity changes, existing-vertex cones and new interior material sites. The check retains the original anatomical owner, mapped source mass and material bounds.

The compiler-side correspondence audit then traced the actual volume of skin from one interface to the other through contiguous source tetrahedra. The shortest confirmed paths among the sampled visible directions at the three witnesses were approximately 2.2 mm, 0.054–0.077 mm and 5.7 mm. The short pelvic patch also passed a direct closest-surface check. These are source-mesh chord distances. Adult regional ultrasound data provide measured comparison values for named sites and should be matched to landmarks before assigning biological thickness. [Regional adult measurements](https://pubmed.ncbi.nlm.nih.gov/27496276/)

Using those correspondences, the installed TetGen compiler rebuilt connected material cavities of up to 358 original skin cells with the original external and internal donor facets fixed. Four of twelve candidates passed geometric checks. All four failed the 1% force comparison on the changed cells alone. In the best torso-associated candidate the energy difference was 0.31% and the nodal force-vector difference was 15.1%, about 0.025 N. The outer pelvic cavity either retained the blocking tetrahedron or changed its prescribed boundary. Original materials and mass mapping remain in the scene.

## Implementation decision

Keep the regional support scene as the current motion comparison. Keep the dense force, strain and physical acceptance gates. Replace the poorly resolved connected layer geometry and validate the boundary tractions, energy, mass and contact against matched native loading. The anatomical structure of the skin/fat interface itself needs correction at these sites. Further isolated tetrahedron deletion and higher Krylov iteration caps lack support from these measurements.

The bilateral thumb-tendon result is in the neighboring forearm evidence directory. The best regional line of action clears the cortical surfaces in all 250 native motion samples, but its transfer still enters the dense skin/fat material. An additional joint fit also failed the common geometry tests. Neither experiment substitutes for the dense whole-body motion run.

Both recovery receipts contain GitHub release URLs and hashes verified by an independent download. They retain the failed attempts, exact compiler inputs for the local cavities, generated volumes, solver records and source programs. The executable audits in this commit consume existing native artifacts and add no simulation state.
