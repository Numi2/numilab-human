## Whole-body tissue and native trial on SSH Numi

The matched control reached 1.56250007e-05 s using [395, 782, 552] Krylov columns across the attempted roots. The new trial reached 1.56250007e-05 s using [395, 901, 288]. Its complete process took 247.11 wall seconds. Both used variable pleural pressure and the same full-body model, tendon bindings and GPU load coupling. The trial rejected at root 2 with diagnostics [2.500000238418579, 2.5, 1460506, 4]. The solver experiment is excluded from the production default.

The native material importer verified 1,457,316 cells and 314,014 nodes. In the reference and opening pose the skin and fat have zero inverted or below-shape cells. Applying the same retained motion to the reconstructed volume leaves 4 skin and 33 fat cells below the shape threshold. The admitted cavities retain reference-material identity and a conservative parent-to-child mass and virtual-work check. The unchanged simulator supplied the saved physical coordinate samples. The new material import has not passed a new coupled motion run.

The six-muscle constrained fit included immutable bones, neighboring muscles, physical skin and fat, source tendon footprints, volume and local tetrahedral orientation. The sum of measured pairwise intersections changed from 85.65358 to 73.05313 cm³ over 21 accepted authoring iterations. This sum may count the same space in multiple pairs. It describes registration in a retained frame. Current control bed force was 11.015 N versus 952.862 N normal weight.

The native source-carrier replay also passed after integrating the appended fat node and six replacement cells. Cell-level source identities and the physical mass of each carrier were conserved. Transferring the six-muscle fit back to the emitted source admitted only 1/16 of the authoring displacement. The independently measured spinal deltoid/fat overlap remained 42.965 cm³.

The complete model has not qualified supported rest, zero material overlaps, continuous breaths and beats, or 0.1× throughput. The measured trial is too short for an integrated steady-state benchmark.
