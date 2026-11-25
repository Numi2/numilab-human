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

## Matched material precision run

Computing bone-local reference points from the loaded body origins before Float32 storage, and using compensated rotation and position accumulation on Metal, reduced the measured self-intersections from 49 to 44 at 8 ms. All 64 physical steps passed. The 11,687 skin–anatomy pairs persisted. The previous regional control had three self-intersections. The new boundary remains experimental.

The complete accepted interval from step 1 to 64 reached 0.00017489×. Including startup, 8 ms took 61.78 s, or 0.0001295×. The relative L2 change in regional bed force was 0.0378% across the 64 steps compared with the same reseated material in the previous arithmetic path. The largest difference in total reaction magnitude measured as the norm of the vector difference was 0.105 N. Fore-aft reaction was still 172.17 N at the endpoint.

A separate distance check sampled matching source triangles on the union of 36 bone and 75 muscle surfaces implicated by the native crossing checks. Each stratum represented up to a share of 512 samples per surface. Absolute winding below 0.2 classified a sampled centroid as exterior, and its nearest point on the skin gave the distance. These sampled maxima are lower bounds on the worst protrusion. The full-mesh crossing checks cover all 878 target banks.

| Matched sample | Previous regional skin | Loaded material with precise transfer |
|---|---:|---:|
| Estimated exterior bone area | 18,713 mm² | 6.29 mm² |
| Largest sampled exterior bone distance | 15.20 mm | 0.232 mm |
| Estimated exterior muscle area | 80,207 mm² | 65,122 mm² |
| Largest sampled exterior muscle distance | 37.30 mm | 40.25 mm |

The abductor pollicis longus surfaces account for the deepest remaining sampled muscle protrusions on both forearms. The two thenar force lines still miss their respective rendered muscle volumes. Their attachments and material points should be reconciled against the same loaded anatomical skeleton before further skin contour edits.

The second native patch applies after registered-bed-material.patch.gz. Recompile the local bed skin Metal library. Keep the separate full respiration library for its other 127 kernels. An attempted startup with the 50-kernel bed library in both positions failed before the first physical step. The independently restored precision archive retains the failed startup and the corrected native run.

## Bilateral abductor pollicis longus force path

The integrated native program applies the revised eight-site paths to actuators 259 and 322 and recalculates distributed tendon tractions at the four original cortical footprints. The radius-to-first-metacarpal shafts pass the registered skin and cortical solid tests at release and both observed moving frames. The maximum Float32 resultant-force error is 1.55e-8 and maximum moment error per unit force is 1.88e-10 m. At the independent native endpoint the amount of route outside the skin drops from 112.42 to 0 mm on the right and from 132.11 to 0 mm on the left.

The path-only candidate remains experimental. At 8 ms neither force route intersects its named muscle belly. About 41 mm of maximum sampled muscle protrusion persists on each side. Regional bed-impulse histories change by 11.44% in relative vector L2 against the matched prior native run. Fore/aft bed force at the endpoint increases from 172.17 N to 229.67 N. Native initialization reports a dynamic release. The final whole-skin audit finds 204 bone, 9831 muscle, 955 vascular and 700 other semantic triangle pairs, and 47 skin self-intersections. Variable pleural pressure was active for 64 accepted 0.125 ms steps. The timed interval achieved 0.000170× and the complete 62.68 s process achieved 0.000128×. The force-path improvement does not establish supported equilibrium, breathing-cycle clearance or long-duration physiological qualification.
