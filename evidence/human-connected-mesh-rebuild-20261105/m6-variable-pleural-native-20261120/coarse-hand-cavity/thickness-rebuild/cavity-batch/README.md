# Refit witnessed skin and fat cavities with shared tangential motion

Run on SSH Numi against the source and exact native Float32 coordinates
pinned at 2664d7d. The prior guarded fit had 16,569 skin/fat–bone triangle
pairs and 96 inside skull vertex–bone pairs. This increment is offline
authoring and keeps the original variable-pressure native runtime separate.

The local cavity screen examined 120 candidate edges and found two
disjoint splits improving minimum shape in both reference and fitted
rest/world frames. Together they cover just 10 of the 428 failed-cell
witness visits. The interior split adds two nodes and splits eight parented
tetrahedra into sixteen. An independent check confirms both signed cavity
boundaries, their enclosed volumes and every parent assignment. The
exterior triangles and bed vertex identities are unchanged by the splits.

The fitter now allows each existing skin–interface–fat column to translate
along two shared tangents as well as moving the three normal coordinates.
The tangents preserve the paired layer thickness by construction. They
have a 2 mm authoring travel bound and keep the current outer/source
objective. The accepted candidate used at most 0.575 mm of tangent
travel. The deformation authoring remains separate from native dynamics.
All determinant, shape, exact containment and thickness checks still apply.

| Candidate | Outer skull | Interface skull | Deep skull | Whole shell against bone |
|---|---:|---:|---:|---:|
| Matched native initial geometry | 702 | 2,311 | 2,685 | 17,878 |
| Previous guarded checkpoint | 506 | 1,350 | 2,533 | 16,569 |
| Two cavity splits plus shared tangents | 483 | 1,253 | 2,523 | 16,439 |
| Same setup with local deep restoration | 480 | 1,244 | 2,504 | 16,408 |

The cavity and tangent change is a combined experiment. In the matched
comparison for local deep restoration, the first three accepted trials
are identical. The original search rejects the fourth and fifth trials.
The restoration variant identifies newly worsened deep source faces from
exact parent-face/bone-triangle pairs and returns their implicated columns
to the previous accepted authoring position. Each restored trial passes
the complete volume, width, mass and contact checks. The two recovered
steps resolve an additional 31 pairs. The elapsed authoring and audit times
are 107.76 s and 95.57 s. This is a one-run comparison of the authoring
path and provides no native GPU throughput measurement.

The final candidate contains 8,346 nodes and 33,000 positive tetrahedra in
both evaluated frames. The independent verifier confirmed 13.93968 kg
total, typed source quotas, receiving masses, density and a maximum donor
route of 68.16 mm under the same 75 mm limit. All width checks pass. The
exact skull audit tests 8,343 shell-boundary vertices and counts 91 inside
pairs versus 96 at the previous checkpoint and 169 in the refined
reference. There are no new inside pairs, no new external-bank crossings
or self pairs and no ambiguous or boundary points. The remaining triangle
count still includes 1,690 newly counted pair identities relative to
the native reference within this skull bank. The full zero-crossing
condition remains unmet.

The residual deep crossings span 133 source faces, with 2,245 of 2,504
pairs in one 121-face edge-connected component. The top twenty source
faces account for 1,292 pairs. This evidence identifies a connected surface
rebuild footprint. Repeating the same interior splits cannot reach most
limiting cells.

The first tangent invocation stopped before a trial produced geometry.
An array of solver variable bounds shadowed the layer-node index array.
The corrected run and the guarded A/B completed. Six tests now cover
scalar plane parity, affine corner bounds, repeated exact queries,
containment and boundary transitions, tangential displacement and its
determinant Jacobian, and parent-pair accounting through local restoration.

The retained native run remains a short release test with all 34 registered
ligament rows. Loaded shoulder behavior, bed equilibrium, the 16-second
zero-crossing run under variable pleural pressure, and end-to-end 0.1×
and 1× performance remain open. No candidate in this increment was
installed in the native scene.

## Reproduction

On SSH Numi, create `cavity-batch-037` next to `skull-contact-036`.
Restore the latter directory as described in its checkpoint. Copy
`rebuild.py`, `fit_layers.py`, `skull_contact.py`, `verify_rebuild.py`,
`test_skull_contact.py`, `solid_skull.py`, `regional_mass.py` and
`audit_solid_vertices.py` into the new directory. Apply `batch-fit.patch`
with `patch -p0` and add `column_motion.py` and `screen_cavities.py`.
Run inside the new directory.

```sh
python3 test_skull_contact.py
python3 screen_cavities.py ../skull-contact-036/sat-continuation-06 cavity-screen.json
python3 rebuild.py batch --whole-skull --all-edges --interior-edge 3263 5829 --depth-contact --restore-local --preserve-solid-and-width --tangent --restore-deep --split-screen cavity-screen.json --resume ../skull-contact-036/sat-continuation-06
python3 verify_rebuild.py --source-checkpoint ../skull-contact-036/sat-continuation-06/rebuild.npz batch
python3 audit_solid_vertices.py batch --all-surface
```

The stored candidates, exact receipts, and traces were recovered from the
private release and checked against their manifest.
