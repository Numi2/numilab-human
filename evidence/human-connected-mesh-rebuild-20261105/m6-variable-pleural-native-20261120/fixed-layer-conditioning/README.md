# Fixed-boundary skin/fat conditioning batch

The reference is the same 311,020 node and 1,448,804 cell mesh admitted by the native full-volume prefix. Freezing all exterior and material interface faces fixes 297,854 nodes. Of 20,085 cells with reference edge-basis condition number at or above 10,000, 13,638 have no movable corner under this constraint. An interior-position-only repair cannot change those cells.

The joint batch screened 88,704 cells with all corners fixed and an edge-basis condition number at or above 1,000, prioritizing the geometric stiffness proxy. It reused the existing 2–3, 3–2 and 4–4 cavity proposals and kept only disjoint same-material operations that reduce the cavity's maximum centered-coordinate condition number by at least 20% and the geometry-only gradient-energy proxy by at least 10%. One screen selected 7,329 operations, reducing the proxy by 7.31% over the whole skin/fat mesh and reducing the cell count to 1,446,015. No coordinates move.

The independent certificate used exact rational arithmetic on the Float32 coordinates. The replacement cavities preserve their oriented boundaries and exact enclosed volume, and every new cell passes the existing absolute volume and normalized determinant floors. The proposal stays offline. The mass carrier, material-parent ancestry and equivalent native forces and strains still need qualification. The proxy improvement is not a measured Krylov iteration or real-time-factor improvement.

Reproduce on SSH Numi with the pinned full-volume 026 launch receipt. Run `prepare_conditioning_flips.py` to reuse the saved cavity-generator source, then run `screen_conditioning_flips.py <launch.json> 100000` and `certify_cavity_replacements.py <launch.json> proposal-all.json` in the existing Python environment.
