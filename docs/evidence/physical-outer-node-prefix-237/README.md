# Direct physical outer-node rendering order

Strict O/I/D volume assembly now reorders combined nodes so the registered O source rows form the mechanical prefix consumed by the existing exact accepted-FEM renderer. It applies a bijective permutation to points, both partition maps, shared I ownership and all subsequent cell/facet maps. Original TetGen node IDs and anatomical source IDs remain explicit. No physical coordinates, cell orientation, material, mass or force equation changes.

The positive prism regression independently permutes the skin and fat TetGen file rows and IDs. The combined result has the exact registered O prefix, unchanged per-cell Float32 point tuples and materials, and exact shared I mapping. Its eight corrupt map cases still reject. The historical incomplete 7-node fixture and existing child semantic-label tests pass. These are compiler tests; no production tetrahedralization, changed volume or motion is admitted.
