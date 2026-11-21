# Volume bounds along the three-dimensional preload step

The restricted offline preload planner estimated a safe step by
extrapolating the volume ratio from its two endpoints. A determinant
is affine only in the normal-only model. With lateral motion it can
cross a guard between valid endpoints, or an endpoint secant can cut
back a step too early.

The new authoring guard evaluates each tetrahedron's determinant as a
cubic in the step fraction. It finds the first crossing of either volume
bound by bracketing on the polynomial's monotone intervals. The bed
clearance remains an exact affine bound. The original constitutive energy,
IPC potential, and convergence checks are used throughout. No native
shader, timestep, material parameter, attachment, or acceptance gate
changes here.

Five tests passed on Apple M6. They cover the cubic reconstruction against
direct matrix determinants, an excursion hidden by valid endpoints, two
competing guards, a small cubic term, and a case where the endpoint
secant over-restricts the step. The existing predictor can be reproduced
by applying the two edits in `predictor.patch` to its pinned predecessor
and placing the new segment module on the existing Python import path.

| Deep-slip trial | Endpoint estimate | Whole-segment determinant |
| --- | ---: | ---: |
| Last converged weight fraction | 2.5% | 2.5% |
| Active bed sites at 2.5% | 101 | 101 |
| Minimum Float32 volume ratio at 2.5% | 0.59342448 | 0.59342448 |
| Rejected weight fraction | 5% | 5% |
| Limiting adipose cell at 5% | 21848 | 21848 |

Converged coordinates agree within 6.02e-11 m. At 5% the exact step
search again stops at the upper expansion bound with a large residual.
At 2.5% all 28,632 authored cells are positive and above the shape floor,
and shell self and cross-layer counts are zero. The changed faces still
introduce nine shell-to-bone triangle pairs relative to the input.
No candidate was installed. This rules out the endpoint secant as an
explanation for the repeated failure to reach 5% in this preload setup.
It does not rule out other changes to physical load transfer. The
combined native motion acceptance remains unqualified.

The command and registered full-precision inputs are the same as in the
prior deep-slip replay. Apply the patch with `patch -o` and use the
patched predictor in that command. This directory includes the minor
source change, its tests, both the retained equilibrated state and the
terminal rejected state, and the independent Float32 audit.
