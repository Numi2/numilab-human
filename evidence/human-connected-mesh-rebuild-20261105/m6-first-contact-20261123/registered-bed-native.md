# Which deficiency limits the supported adult?

The full-volume reference starts with 10,764 transverse outer-skin intersections against internal anatomy. The starting scene, native skeleton, physical outer-skin triangle ordering and reference asset match the release witnesses. Of those pairs, 10,754 persisted to the four-step endpoint. Seven new pairs appeared and ten cleared. Correcting the motion solver alone cannot repair this input anatomy.

The 2,713 affected skin triangles form 104 edge-connected patches. The largest patch has 727 pairs and spans the left deltoid and supraspinatus surfaces. Another has 685 pairs against the left brachioradialis, and a third has 567 across multiple left forearm muscles. The recovery archive records the exact physical nodes, triangles and participating structures for all 104. Triangle count and touched triangle area are unsuitable estimates of displaced tissue volume.

## Native comparison

Both trials ran four complete 15.625 µs steps on SSH Numi with the same binary, source, full physical skin/fat mass, bed, gravity, variable pleural pressure and nonlinear acceptance checks. The experimental solve inserted six displacement modes from the certified reference bed patch into Matter's existing current-operator coarse correction. The six modes represented 76.7% of the squared displacement in the 19 reference responses. The all-port error bound applies to the full localized solve. The six-mode approximation has a different accuracy.

The installed full coupled row scaling refused the experimental coarse correction during native preparation. The paired test used the runtime's admitted unscaled configuration on both sides and retained the scaled refusal in the archive.

| Four-step observation | Unscaled control | With six bed modes |
|---|---:|---:|
| Accepted physical time | 62.5 µs | 62.5 µs |
| Active Krylov columns by step | 111, 113, 105, 162 | 111, 113, 105, 162 |
| Body runtime | 42.86 s | 54.21 s |
| Cold end-to-end wall time | 184.87 s | 196.40 s |
| Invalid material cells | 0 | 0 |
| Intersections between or within the three skin/fat layers | 0 | 0 |
| Outer skin against internal anatomy, triangle pairs | 10,761 | 10,761 |

The GPU performed 491 nonzero coarse corrections and reported 64 valid factorizations in the experimental run. Body time increased 26.5% in this pair with identical iteration counts. Maximum material position difference was 2.32 nm. Recorded lung volume, blood volume, ventricular volumes, blood gases and pleural pressure agreed across the four steps. This interval establishes neither a cardiac cycle nor a respiratory cycle. The terminal bed reaction was approximately 15 µN for a body weight of 952.9 N.

The production solver keeps its current settings. The archived implementation permits reproducing this negative experiment. Fixing the connected skin/fat registration around the measured muscle and vessel geometry takes priority over further release-bed Krylov tuning. A replacement must preserve routed mass and the physical load path and pass the native crossing and material audits at release and through motion. The separate compact scene's prior 16-second motion does not establish these properties for the full tissue scene.

[Native measurements](registered-bed-native-comparison.json) · [Connected skin regions](registered-crossing-regions.json) · [Verified recovery and native frames](registered-bed-native-recovery.json)
