# Native skin and fat volume registration

The production registered-volume importer and full 86-binding role consumer on SSH Numi now accept the rebuilt 1,450,187-cell tissue volume. The exact original-format inputs and a matched run of the same importer on the previous volume are retained. [Native comparison](material-joint-native.json) · [Independently restored inputs and source](material-joint-recovery.json)

| Registered quantity | Verified result |
|---|---:|
| Occupied physical volume nodes | 311,361 |
| Newly generated interior points | 447, including 106 reused interior index slots |
| Registered outer skin, inner skin and deep fat nodes | 297,854 |
| Registered material facets with exact incident elements | 832,928 |
| Authored head-wall faces with corrected incident cell and original source lineage | 180 |
| Deep attachment requests | 14 |
| Body weight bindings checked by the role consumer | 86 |
| Maximum world-coordinate reconstruction error from the weights check | 8.45e-17 m |

The repaired connectivity left five nodes with source donor/region labels absent from their current incident cells. For those nodes only, the numerical incident-cell selection uses the largest incident mass contribution in the same anatomical region. The force-frame donor, historical source parent, barycentric map, rigid binding and physical mass are preserved. Each new interior point has an independently identified historical parent in its registered material, donor and region, with maximum projection distance 14.38 mm under the existing 20 mm bound. Its operational map uses the actual vertex of an incident emitted tetrahedron. The complete importer checks positive geometry, reciprocal shared interfaces, source identity and attachment mappings. [Joint source maps and witnesses](material-joint-registration.json)

The prior facet compiler updated incident tetrahedron IDs in two fields but left the 180 derived wall records pointing to the previous mesh in a third. The native reader rejected that earlier candidate. The corrected compiler verifies each wall against its pinned source-face ID and the actual new cell label, then writes the new incident cell consistently. [Facet mapping](material-joint-facets.json)

## Scope of the result

This is the complete input registration for this volume. The physical mass totals were qualified separately. The native time-stepping scene still requires the carrier-to-cell allocation for this exact topology, a verified loaded equilibrium, and motion testing. This import does not establish zero skin-to-tissue intersections or real-time performance. [Material mass results](forearm-current-material.md) · [Coupled motion and supported-bed findings](shoulder-physical-motion.md)

To run the same native importer in the prepared Numi workspace, use the retained source, exact files and parent launch named in the recovery manifest.

```sh
python3 verify_volume_install.py "$parent_launch" "$native_pose" "$rebuilt_mesh" "$registered_surface_check" "$joint_map" "$facet_map" "$new_output"
```
