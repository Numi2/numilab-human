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

## Initial hand contact confirmed

The separate three-step endpoint capture completed normally with matching
body and physiology clocks. The independent constitutive audit uses the
existing registered material parameters and the exact split displacements.
It reproduces the native deformation determinants to within 5.0e-10 and
the witness-node material force within 1.24e-6 N. At local interface row
4672 the skin supplies +27.56 N along the adverse motion axis, and the
adjacent fat supplies −0.18 N.

The first-step force balance places the original load at outer row 2261
(compiled node 3927). Three independent closed-solid rays place it inside
registered bone 114 on body 109, at 1.665 mm nearest distance. The registered
unilateral force of 30.01 N matches the native step's nonbulk force to
within 0.0018 N. Outer row 2262 also lies 0.922 mm inside bone 108 on body
107. The two audited deep-fat points are inside bone 108 by 2.133 and
1.854 mm. The connected internal fit preserves these initial outer/deep
geometry defects. Moving the internal interface again would leave the
source of the release transient in place.

There is a separate bed initialization issue. The first Matter contact
assembly adds a 0.21532 N·s trial impulse at the single outer node with
24 nm initial bed clearance. This is an initial residual contribution.
It does not measure the accepted physical support reaction. The accepted
trace continues to report zero support impulse. The fourth-step limb
contact diagnosis and the bed equilibrium problem have separate owners.

### Reproduce the load attribution on SSH Numi

~~~sh
python3 run_hand_trial.py --connected --capture-three
python3 audit_native_endpoint_force.py connected-interface-replay-capture-three
python3 run_hand_trial.py --connected --contact-rhs
python3 build_hand_trial.py --witness
python3 run_hand_trial.py --connected --solid-witness
python3 compare_initial_hand_load.py
~~~

The full-bank audit is a preparation diagnostic. It uses the same loaded
bone banks and independent inside/outside ray tests. It performs no
per-step physics and does not alter any contact rows, tolerances, mass or
constraints. The four witnessed rows are a localized audit.
