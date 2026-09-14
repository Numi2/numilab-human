# Full skin shell candidate — 2026-09-14

The Human source now carries a reproducible, hash-bound exterior skin shell
candidate. It uses BodyParts3D 4.0 member `FJ2810`, selects the exact outer
connected component from the compound skin solid, and registers four visual
body influences per retained vertex against the source-bound MyoSim rest frame.

The source mesh contains 102,467 vertices and 203,382 triangles. The retained
outer shell contains 54,949 vertices and 109,183 triangles, with 86 registered
body influences and a maximum rest-pose reconstruction error of
`1.3383253895372864e-15 m`. The native payload is `NHSKIN1` ABI 4 and its
payload hash is recorded in the immutable receipt.

Reproduce the receipt with:

```sh
numi human skin-shell-candidate \
  --output Docs/media/skin-shell-candidate-20260914/receipt-v1.json
```

The receipt is [`receipt-v1.json`](media/skin-shell-candidate-20260914/receipt-v1.json).
The exact visual payload and 185-anchor registration are retained beside it:
[`bodyparts3d-myosim-skinned-shell.nhskin`](media/skin-shell-candidate-20260914/bodyparts3d-myosim-skinned-shell.nhskin),
[`bodyparts3d-myosim-skinned-shell.manifest.json`](media/skin-shell-candidate-20260914/bodyparts3d-myosim-skinned-shell.manifest.json),
and [`registration.json`](media/skin-shell-candidate-20260914/registration.json).

This closes source identity and visual registration for the skin domain only.
It does not assign shell thickness, physical volume, mechanical mass,
constitutive material, collision/contact, self-contact, muscle sliding,
adipose geometry, or subject calibration. The pinned source inventory has no
adipose member, so fat remains an explicit missing source rather than an
inferred layer.
