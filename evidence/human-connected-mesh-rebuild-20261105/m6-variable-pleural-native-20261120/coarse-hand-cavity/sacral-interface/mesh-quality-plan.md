# Remove small cutter features before exact re-emission

The first complete paired authoring surfaces clear all tested crossings.
The fixed-facet tetrahedralization retains positive determinants but
produces 142 and 159 cells below the existing quality floor, with edges
as short as a Float32 spacing. The cutter union contains many small
features introduced by dilation and Boolean intersection.

Use the retained solid simplifier on the cutter before it reaches the
source muscle. Test a 50 micrometre reduction tolerance, below the
100 micrometre geometric margin. Keep the source meshes exact and
rerun the full independent crossing checks. Compare fixed-facet
TetGen with and without coplanar merging on the original and simplified
cutter. Accept volume geometry only when the emitted boundary matches the
authored triangulation, volume closes and all cells pass existing gates.
