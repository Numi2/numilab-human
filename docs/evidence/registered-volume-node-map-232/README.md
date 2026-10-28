# Registered surface node map regression 232

The native-domain-217 adapter now records exact registered O, I and D source-surface vertex rows, shell node rows, source face triplets and shell face rows. The combined TetGen node map appends the existing CSV columns registered_surface_layer and registered_surface_vertex_row; production mode requires exact one-to-one O/I/D maps and rejects missing, duplicate, collapsed, or role-mismatched maps. Only the explicitly registered I rows are merged across skin and fat.

The assembled-shell fixture verifies face-role and ancestry row correspondence for O, skin-I, fat-I and D. Its retained legacy 7-node/2-tet import control still passes and explicitly reports its O/D node map as incomplete. The additional positive triangular-prism fixture verifies 9 combined nodes, 6 positive Float32 cells, reordered TetGen IDs, exact shared I mapping and 8 fail-closed O/D map cases.

The conditioned input review reads the retained compact O and head collar: 32,002 O points, 63,959 O faces, 67,296 typed ancestry entries, exact O/collar points and faces, and 45 O/I/D head-ring rows with 90 skin and 90 fat wall parent facets. The retained conditioned input packet does not contain full production I/D surface faces, so those real-data face maps are not claimed as checked; exact O/I/D assembly mapping is covered by the bounded fixture.

All results are adapter or source-input checks. No production TetGen, child volume, mass qualification, physical admission, or motion result is claimed.
