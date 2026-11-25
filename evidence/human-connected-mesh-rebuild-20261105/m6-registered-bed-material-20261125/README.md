# Registered skin and fat in the native regional bed scene

The native M6 accepted 64 steps of 0.125 ms using the loaded material boundary for weighted skin support, contact Jacobians, bed section compliance and presentation. The shoulder certificates cover six bilateral cortical paths in this exact seated pose. The default regional scene remains the control.

| Measured at 8 ms | Previous regional skin | Registered material boundary |
|---|---:|---:|
| Skin–bone triangle pairs | 2,856 | 212 |
| Skin–muscle triangle pairs | 8,189 | 9,822 |
| All skin–anatomy triangle pairs | 11,672 | 11,687 |
| Skin self-intersections | 3 | 49 |
| Final bed reaction, vertical | 945.3 N | 938.9 N |
| Final bed reaction, head-to-foot axis | 214.5 N | 172.2 N |
| Elapsed time including startup | 61.49 s | 61.95 s |

**Disposition.** The registered boundary is experimental. It encloses the original right thumb force segment, which previously protruded 6.07 mm. Both thenar force segments still miss their respective rendered muscle volumes. The skin surfaces have different triangle tables, so pair counts describe each emitted mesh and do not measure protruding tissue volume.

**Registration and forces.** All 76,153 outer nodes correspond exactly to the loaded material coordinates. All 152,261 oriented outer faces match the material wall, including source row ownership. Each of the 86 body lanes retains its material weights. Using the actual opening skeleton avoids a 7.52-degree reference frame discrepancy. Each bed-facing vertex contributes its own skin and fat thickness to the existing compliance calculation. The reduced owner represents through-thickness normal compression at the bed. Motion of the remaining detailed skin uses the skeletal transfer. The whole-volume skin/fat FEM was inactive in this run.

**Loading and qualification.** The native bed selected six loaded regions after a one-time 1.740 mm translation toward the plane and a 0.151-degree rotation of the body. Constitutive preload was 954.01 N against 952.86 N required. Every step retained the existing physical acceptance path, including 26 fascia regions, 34 registered ligament force rows and variable pleural pressure. Initial recruitment reported unbalanced. The timing interval from steps 1 to 64 reached 0.000174×. The run did not span a heartbeat, breath or the 16-second qualification.

**Reproduction on SSH Numi.** The independent recovery archive includes the exact native launch records, rejected pose, matched source and binaries, contact receipts and raw replay captures. The owner delta lists both sides of the application and contact change. Apply the compressed patch under tools/native_ligament_contact/registered_material_volume/regional_bed to the matching native owner. The compiler runs with the existing NHSKIN v5 reader on the Human tools Python path. Its positional arguments are the dense native launch JSON, its original NHSKIN and a new output directory. Set NUMI_HUMAN_REGISTERED_BED_MATERIAL_RECEIPT to the resulting material-boundary.json for the regional bed run. The cortical include in this directory belongs to the seated launch.
