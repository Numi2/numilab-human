# Exact interior cut-loop triangulation

The common atrial/ventricular material boundary exposed closed interior loops in source triangles that also contain boundary-connected cuts. The prior connected-only triangulator rejected these valid planar arrangements; its separate single-loop fallback also rejected the combined case.

The existing face owner now inserts two visible, noncrossing diagonals between distinct existing vertices of each disconnected closed component and the boundary-connected graph. It then uses the existing exact face traversal, triangulation, source-area, constraint-edge, winding, incidence and T-junction checks. No coordinate or original cut is removed. Dangling and nonsimple cuts still fail.

On the SSH Mac mini, all 3,512 affected source faces pass. Twenty formerly unsupported cases are retained as source-bound local fixtures; 14 tests pass, including the previous 42-intersection cardiac regression. Rights and inferred-reference provenance accompany the fixture. The complete common material surface and native cardiac motion require their own checks; local face triangulation does not establish those outcomes.
