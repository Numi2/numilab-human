# Skull shell thickness and mass redistribution

The baseline is the exact native initial pose from the matched source audit at
0d05ab99. Its skull bank has 702 outer crossings and 5,698 crossings across the
whole shell. Both the bin and the source coordinates are hash pinned.

Test whether permitting local transport of the same anatomy and material mass
allows the inner layers to fit the measured source-to-bone clearance. Preserve
the native density intervals, the per-source mass constraints, and the bed
coordinates. Route within 75 mm and only within a single material/donor pair.
Keep every edited tetrahedron at or above the minimum determinant and shape
of the entire unshifted refined mesh. Each candidate will be screened after
Float32 emission.

Test the existing patch footprint and then the footprint selected from all
three layers. Compare against the previous independent-layer proposal and a
geometry-identical bisection. The expectation is greater thickness correction
and fewer deep skull pairs without self-intersections or density failures.
Require zero skull pairs, zero new external bone pairs and a certified
conservative route before carrying the geometry into a native build.
Limit each fit to eight linearizations and each allocation to 25 seconds.
