# Connected hand interface repair on Apple Metal

The coarse parent has 7,233 nodes and 28,632 tetrahedra. Two fat cells next to a free hand interface node restricted the third physical step. A boundary-preserving retriangulation screen evaluated 87 flips and found no admissible move touching either witness. Flipping their shared face alone overfills the cavity.

The replacement fits the common skin/fat interface in the full incident element star. The single-node trial shifts the interface by 16.54 µm. The connected trial adjusts ten interface points by at most 77.35 µm and covers their 184 incident cells. It keeps the external and deep boundaries, connectivity, owner identities and both material volumes. The reference fit checks incident determinants along the displacement path and explicitly rounds emitted coordinates to Float32. Both native builds use the unchanged material validity, contact and coupled residual gates.

| Configuration | Third-step mechanical norm | Accepted steps | Accepted time |
|---|---:|---:|---:|
| Pinned parent control | 1.5746e-3 | 2 | 31.25 µs |
| One interface node | 6.9441e-4 | 2 | 31.25 µs |
| Connected interface | 8.2161e-6 | 3 | 46.875 µs |

The comparison uses the same 1.25e-5 domain mechanical threshold. A separate repeat reproduced all three accepted steps and the fourth-step failure. Its first full correction crosses the existing upper determinant limit at the same two hand fat cells, global rows 29419/29516 and parent rows 26104/26201. Its final mechanical norm is 5.7151e-4, while the vascular norm remains inside its limit. The rejected step publishes no accepted physiology row.

The connected candidate keeps the prepared whole-body mass at 97.131950691109523 kg and the declared skin/fat debits at 3.4525707761420956 kg and 10.487108964465744 kg. The cooked tissue sum changes by 7.1e-15 kg. The uploaded FEM sum changes by 1.55e-9 kg. The native runtime binds the same 34 registered ligament rows. These are preparation and short replay checks. Shoulder loading through supported motion remains unverified.

The 28.37-second process run measures 1.65e-6× from launch. This is a diagnostic release. Its peak initialization acceleration residual is 575.59 and every accepted step reports zero support impulse and root assistance. Pleural pressure stays at -490.332489014 Pa at this time resolution. The run does not qualify loaded equilibrium, whole-body crossing clearance, 16 seconds, sustained changing pleural pressure or 0.1×. The separate full-resolution reference repaired 125 inverted and 154 low-quality tets; this coarse trial does not install a new state on that mesh.

## Reproduce on SSH Numi

Use the archived source and asset paths in the launch records. The scripts belong in the retained hand trial directory, next to the pinned parent evidence and runtime overlay. To reproduce the connected construction and trial from a fresh workspace with those input paths, run:

~~~sh
python3 fit_connected_interface.py
python3 build_hand_trial.py --connected
python3 run_hand_trial.py --connected
python3 run_hand_trial.py --connected --trace-fourth
~~~

The two native patches and compile receipts preserve the exact observed builds. The default builder produces the same one-node edit through the newer shared overlay path. The small proof scripts use NumPy and SciPy in the existing Numi prep environment. The attached launch records and compact output preserve process status, accepted clocks, residuals and physical limit witnesses.
