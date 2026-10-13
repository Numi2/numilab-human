# Exact positive-ray candidate filtering

The cardiac partition certificate uses 64 deterministic parity rays whose XYZ
components are strictly positive. After the unchanged exact boundary check, it
filters out faces whose closed AABB upper bound is below the query point on any
axis. Such a face cannot meet any configured ray at positive time. Equality and
original face order are preserved. Exact rational slab, plane, edge/triangle
predicates, ambiguity retries, the 64-ray bound, and all certificate admission
rules remain unchanged. An empty candidate list yields outside.

Validation passes 23 tests and 256 subtests. Differential controls compare the
unfiltered exact loop across rational query grids, negative translated/oblique
closed surfaces, boundaries, edges, coplanar first-ray retries, and an empty
upper orthant within global bounds. Existing coverage, intersection, volume and
partition-certificate controls also pass.

A matched same-process CPU fixture measurement used 24 disjoint tetrahedra
(96 faces) and 24 interior queries in five alternating-order pairs. Locations
were exactly identical. Candidate face rows fell from 2,304 to 430. Median
query-batch time was 0.027399167011026293 seconds for the full loop and
0.008987792010884732 seconds with the filter. This bounded Python exact-Fraction
fixture measurement is not a full anatomy, Metal or application benchmark.
The machine-readable receipt includes both source/reference SHA-256 hashes.

The optimization does not prove source anatomical identity, a cardiac material
partition for a real specimen, or physiological behavior. Those source-bound
certificate gates remain required.
