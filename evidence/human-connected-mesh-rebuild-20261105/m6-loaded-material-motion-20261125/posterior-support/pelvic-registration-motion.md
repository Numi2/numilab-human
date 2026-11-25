# Pelvic skin and fat registration under native motion

The dense control starts in dynamic release. Its source-backed FEM bed carries about 11 N against 952.86 N of weight. All comparisons use the existing conditional contact, anatomical joint stops, variable pleural pressure and accepted-state ownership on SSH Numi, Apple M6, 16 GB, macOS 27.0 (26A425).

| Terminal state after 15.625 µs | Pelvic overlap (mL) | Wall–cortex pairs | Low-shape cells | Principal-stretch failures | Bed force (N) | Body wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Control | 2.915401 | 6,568 | 40 | 506 | 10.992 | 85.62 |
| Force-limited fit | 2.886135 | 6,571 | 48 | 517 | 10.983 | 91.42 |
| Connected source owners | 2.891909 | 6,568 | 40 | 501 | 10.981 | 92.31 |

Moving the two occupied pockets with surrounding tissue removed 0.346 mL in the opening pose. The native determinant checker rejected two cells. Changing displacement encoding by at most 2.15e-11 m passed all 1,457,778 executable intervals. The larger edit still increased one nodal force by 590.53 N and the native equilibrium solver rejected its first step. Moving only the adipose portion without fitting the shared interface inverted 174 incident cells.

Limiting the change in the existing tissue forces produced a two-step native release. A 70 mm spatial neighborhood also moved a left upper-arm deep-fat vertex by 21.5 µm. Three new cortical crossings involved the same humeral triangle. The occupied source cells belong to the pelvis, torso and L5. Traversing mesh edges alone still reached the upper arm. Combining an edge-distance bound with those three source owners retained 2,614 connected candidate nodes. Current coordinates are the only material quantity edited. The native mass, stress-free geometry, mechanical contacts and constitutive laws retain their owners.

The connected trial added 0 cortical triangle pairs and removed 0. The strain comparison recorded 2 newly failing cells and 7 recovered cells. Matched cell masses are 3.452571 kg of dermis and 10.487108 kg of adipose. Its skeletal acceleration along the upward bed normal was -9.79833 m/s².

The two newly failing dermal cells have reference altitudes of **0.0369 and 0.1975 µm**. None of their vertices moved during preparation. In the native captures, the largest node displacements relative to each cell's mean displacement differed by 0.147 and 0.705 µm between the runs. Those differences exceed the corresponding altitudes. Source-conditioned remeshing and conservative mass transfer are warranted at these cells before accepting this registration.

A rigid support-plane calculation also tested whether simply changing the supine orientation would give the skin a larger initial footprint. For the represented skeleton and skin/fat mass, the nearest stable lower-hull facet differs from the existing bed normal by only **0.002 degrees**. Both orientations place approximately 24 cm² of projected outer skin within 2 mm of the plane. This geometric observation does not supply a reaction force.

The control remains the published default. The experimental runs do not demonstrate settled support, a whole heartbeat or breath, a 16-second crossing-free trajectory or sustained 0.1× operation. Use the [matched numeric results](pelvic-registration-motion.json) and [independently verified native archive](pelvic-registration-recovery.json) to replay the experiments. The archive includes the exact source proposals, failed native trials, complete terminal captures, current geometry, executable and traces. The unchanged external assets are identified by the parent launch manifests.

The generic force check runs in the preparation environment on SSH Numi:

```sh
v=tools/native_ligament_contact/registered_material_volume/regional_bed
"$HOME/numi-human-prep-venv-20261005/bin/python3" "$v/audit_registration_forces.py" "$CONTROL_DIR" "$PROPOSAL_NPZ" "$OUTPUT_JSON"
```
