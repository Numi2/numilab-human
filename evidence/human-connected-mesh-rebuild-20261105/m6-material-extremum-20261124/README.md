# Native tissue search-direction experiment

The signed volume of a tetrahedron changes cubically along a Newton segment. The experimental GPU calculation uses the determinant and its derivative at the most violated point of the segment. It uses the stationary-point condition to distinguish an extremum very close to the endpoint in Float32. The original material equations, permitted deformation interval, line search and fresh physics certificate were used.

The Apple-Clang numerical comparison against independent double-precision cubic extrema and perturbed-endpoint derivatives produced `segments=26279 interior_extrema=2536 max_value_error=1.25158e-06 max_gradient_error=1.04336e-06`. The same function compiled with the native Metal shader.

The full M6 scene with variable pleural pressure accepted 2 physical steps (1.5625e-05 s). The cold process took 247.331 s. The matched three-step control used the same executable, model and launch options with the production Metal library. Both builds accepted the first two steps. On the third step the production GPU used 552 Krylov columns and rejected equilibrium with a residual of 0.0318247. The experiment used 352 columns on the third step and rejected a skin candidate with material J of 2.5000002384 against the configured 2.5 maximum. A zero residual field attached to this geometry rejection has no convergence interpretation. The control took 256.852 s cold. Neither trial completed the third step. No steady-state speedup was measured. Two interrupted control launches are retained with their disk-reserve records.

The executable source, Metal library, launch, outputs and tests are in the independently downloaded recovery archive. The production solver and anatomical data were preserved.
