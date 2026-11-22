# The loaded right shoulder

The highest torso load sits over the right scapular region. In the accepted dense frame, four loaded skin nodes account for 66.5% of the replayed elastic bed reaction. The first four rows are the four largest loads. The fifth had one of the largest changes since initialization. The table measures distances in the actual native pose from the deep-fat boundary immediately above each listed load. Distances to soft tissue and cortex are unrestricted three-dimensional closest-triangle distances across the full registered anatomy. The fourteen existing elastic attachment positions come from the verified native executable's source table and the registered donor sidecar.

| Bed node (zero-based) | Static elastic contact (N) | Closest other soft surface | Closest cortex | Nearest registered deep attachment |
|---:|---:|---|---|---:|
| 23186 | 849.62 | Right teres major, 2.46 mm | Right scapula, 32.91 mm | 254.47 mm |
| 24331 | 309.46 | Right infraspinatus, 10.66 mm | Right scapula, 33.47 mm | 257.21 mm |
| 24247 | 216.27 | Ascending right trapezius, 8.26 mm | Right scapula, 34.51 mm | 252.22 mm |
| 24538 | 117.01 | Right gluteus maximus, 6.88 mm | Sacrum, 42.63 mm | 325.05 mm |
| 24013 | 115.52 | Right gluteus maximus, 6.25 mm | Sacrum, 45.13 mm | 298.64 mm |

The earlier muscle-envelope-only scan reported no muscle on the vertical path above vertex 23186. The registered thoracic payload also contains muscle. Tracing all 878 admitted banks found the external, internal and innermost intercostals from the existing respiratory model on that ray. Their disjoint union covers 9.27 mm of the 72.48 mm between the deep-fat boundary and the first vertical cortical hit. The nearest surface in three dimensions is teres major. These source identities, distances and witnesses are retained in [the accepted-frame measurements](conserved-material-peak-paths.json). Distances alone do not identify an active contact or measure a biological gap.

The native log confirms fourteen finite elastic attachments, zero hard ties and no added attachment mass. Other contact forces can also reach the skeleton. The regional decision is to track the load through deep fat, the actual posterior shoulder tissues and the scapula with the existing articulated and muscle owners during loaded initialization. The same bed patch must maintain a consistent resultant and moment when its contact sampling is refined. Forcing a distant attachment or retuning a ligament to compensate for bed load has no support in this measurement. The matched control's shoulder motion and ligament function remain unqualified.

To reproduce, restore the registered native frame and inputs via [the verified manifest](bonded-recovery.json). Use the same SSH Numi environment as the native replay. The peak-load option also includes the four largest endpoint forces. The default muscle-only and force-replay outputs reproduce their previous captures byte for byte. Exact triangle witnesses, source hashes, compiled attachment rows and the verified frame identify every reported distance.

```sh
"$PY" "$u/publication/tools/native_ligament_contact/registered_bed_work/audit_dense_bed_response.py" "$u/conserved-two" "$u/paired-reference" "$u/conserved-transfer.json" --peak-loads > "$u/conserved-bed-peaks.json"
"$PY" "$u/publication/tools/native_ligament_contact/registered_digit_material/measure_dense_support_paths.py" "$u/conserved-two" "$u/conserved-bed-peaks.json" "$W/resting-surface-clearance-001/audit.py" --all-banks
```
