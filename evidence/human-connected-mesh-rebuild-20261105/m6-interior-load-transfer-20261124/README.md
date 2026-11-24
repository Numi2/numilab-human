# Native bonded tissue transfer and bed-load check

Moving two material-interior sampling points improved skin cell quality through two accepted steps of the complete variable-pressure Human scene on Apple M6. Both positions were embedded using the same source barycentric coordinates in the reference and loaded meshes. Boundaries, cortical contacts, attachments, tissue identities and tetrahedron connectivity are unchanged. Material mass was reintegrated by cell overlap. The native skin and fat source-carrier allocation replays passed.

Reintegration touched 68 skin cells. In 42 of them the historical source compiler recorded multiple contributing faces in a single group identity. The old reconstruction tested that identity only as a literal face. The old allocator also chose the first contributing face when it could not find a literal match. The producing compiler, reconstruction check and allocator now share the source encoding. All 68 cells retain the original source group and the same set of contributing faces. The source labels remain byte-identical.

| Matched native endpoint | Control | Remapped |
| --- | ---: | ---: |
| Accepted time, μs | 15.625 | 15.625 |
| Skin cells below shape threshold | 6 | 4 |
| Fat cells below shape threshold | 34 | 34 |
| Inverted cells | 0 | 0 |
| Intersections among four material surfaces | 0 | 0 |
| Bed normal force, N | 11.01472 | 11.01471 |

The two local forces on matched virtual displacements differ by up to 0.306%. The whole-mesh principal-stretch comparison reports 0 failures. Regional masses are 3.452571283 kg skin and 10.487108427 kg fat. The native contact audit across 185 bones still gives 5.9764 cm³ in summed pairwise cortical intersection. Pleural pressure coupling was enabled. The capture is too short to qualify ventilation or circulation.

The native body ran at 2.186e-07× with variable pleural pressure. Including compilation-independent launch preparation and capture, 1.5625001e-05 s took 223.67 s (6.986e-08×). The scene initialized as a fresh release. It supplied 11.015 N of 952.862 N normal weight at the measured step. Sixteen skin nodes carried the terminal contact impulse. The estimated gravity line falls 28.3 mm outside their support polygon. The estimate combines the articulated mass and remapped tissue. The remaining 0.476 kg could move its location by at most 4.8 mm if it is within the registered tissue radius.

The new three-step request accepted 2 steps. Its fresh mechanical residuals were [2.31414306e-05, 7.72473577e-05, 0.031719178]. On the matched pre-remesh third attempt the per-object observer attributes 99.99979% of the squared mechanical residual to bonded skin/fat. Changing the per-tile material witness priority produced the same third-step residual and active Krylov counts as control. The tested production predicate and rollback remain active.

The next integration target is a gravity-loaded starting pose whose distributed bed reactions balance weight and moment through the registered tissues and skeleton. A long release replay cannot qualify supported rest while its starting loads are unbalanced. The full 16-second clearance and performance gates remain open.
