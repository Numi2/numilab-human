# Native skin-shell visual admission — 2026-09-14

The native-compatible v2 payload was run through the current equilibrium
visual probe on the physical Apple M4 Pro. The run used the source-bound
157-body NHRIGID artifact, the 416-route muscle artifact, the official
185-bone `NHBONES1` payload, and the v2 `NHSKIN1` shell payload.

Reproduce the receipt from the captured native logs and frames with:

```sh
numi human skin-shell-native-visual \
  --output Docs/media/skin-shell-native-visual-20260914/receipt-v1.json
```

The native receipt records an empty stderr, the exact stdout transcript, the
v3 visual manifest, the MRVPACK2 visual pack, and four 512-pixel frames. The
skin-shell pixel counts were:

| view | positive shell pixels |
| --- | ---: |
| front | 29,661 |
| oblique | 25,803 |
| side | 18,454 |
| rear | 30,666 |

The native summary reported `metal_pose_device="Apple M4 Pro"`,
`renderer_device="Apple M4 Pro"`, `core_bodies=157`,
`bodyparts_skin_shells=1`, and
`skin_shell_binding=four_body_registered_source_bone_surface_local_linear_blend_world_rest_normals`.
The visual pack content hash is
`2ff21f248c52f6e87cecb2ed0691ff240a1d86f16cca16ece261eb12a3502d1a`; the
file hash is `6889f7cfa7d1899116e7530021c53a554b0d29a7927109662e26b8d3f8d2ad04`.

The result is a qualified native visual admission only. It proves that the
source-compatible shell renders through the source pose and official bone
registration. It does not prove shell thickness, tissue volume, mass,
material, collision, contact, deformation, adipose geometry, dynamic
mechanics, standing, walking, or subject calibration.
