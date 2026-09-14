# Full skin shell candidate — 2026-09-14

The Human source carries a reproducible, hash-bound BodyParts3D 4.0 exterior
shell candidate for the one adult male source package. It uses member `FJ2810`,
selects the exact outer connected component from the compound skin solid, and
registers four visual body influences per retained vertex against the
source-bound MyoSim rest frame.

The source mesh contains 102,467 vertices and 203,382 triangles. The retained
outer shell contains 54,949 vertices and 109,183 triangles, with 86 registered
body influences and a maximum rest-pose reconstruction error of
`1.3383253895372864e-15 m`. The native-compatible `NHSKIN1` ABI 4 payload is
bound to the official `NHBONES1` registration fingerprint `6a48e223`.

Reproduce the current source receipt with:

```sh
numi human skin-shell-candidate \
  --output Docs/media/skin-shell-candidate-native-v2-20260914/receipt-v2.json
```

The current receipt is
[`receipt-v2.json`](media/skin-shell-candidate-native-v2-20260914/receipt-v2.json).
Its native-compatible payload, manifest, and checksum are retained in
[`skin-shell-candidate-native-v2-20260914`](media/skin-shell-candidate-native-v2-20260914/).
The official source bone manifest remains
[`nhbones1.manifest.json`](media/numi-human-lower-joint-focus-v1/receipts/nhbones1.manifest.json).

The first v1 candidate is retained as historical evidence in
[`skin-shell-candidate-20260914`](media/skin-shell-candidate-20260914/). Its
source geometry and registration were valid, but its `NHSKIN1` header carried
`d33ad0d2` while the native `NHBONES1` source carries `6a48e223`; the native
probe therefore rejected it. v2 changes only that compatibility binding and
keeps the v1 artifact immutable.

The native M4 Pro four-view admission is documented in
[`SKIN_SHELL_NATIVE_ADMISSION_20260914.md`](SKIN_SHELL_NATIVE_ADMISSION_20260914.md)
and its immutable receipt is
[`receipt-v1.json`](media/skin-shell-native-visual-20260914/receipt-v1.json).

This closes source identity, native registration compatibility, and visual
render admission for the skin domain only. It does not assign shell
thickness, physical volume, mechanical mass, constitutive material,
collision/contact, self-contact, muscle sliding, adipose geometry,
deformation, or subject calibration. The pinned source inventory has no
adipose member, so fat remains an explicit missing source rather than an
inferred layer.
