# Exact vessel body-frame links — 13 September 2026

The new `organ-vessel-body-links` owner consumes the six-vessel source-frame
receipt and the hash-bound MyoSim full-body reference manifest with its sibling
`NHRIGID2` payload. It validates the payload header, source/core mapping table,
all 103 source-body poses, and every packed pose against the manifest before
emitting the immutable [body-link receipt](media/organ-vessel-body-link-20260913/body-links.json).

The receipt binds the exact source members to the compiled source/core bodies:

| vessel members | MyoSim body | source body | core body |
| --- | --- | ---: | ---: |
| `FJ3413`, `FJ3411`, `FJ3427`, `FJ3645` | `torso` | 9 | 20 |
| `FJ1932`, `FJ3441` | `Abdomen` | 4 | 7 |

The authoritative manifest is SHA-256
`2cea81e64a3c8da63731ae67c40ddc39aa61d9c4d0436000e806d25005e5b6d8`; the
`NHRIGID2` payload is 60,324 bytes with SHA-256
`6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44`. The
receipt itself is SHA-256
`043981486eb4353129b50e40fa2db26f83a9a12370a6c9c1d23096e7ce2855a6`.
The complete artifact directory is covered by
[`SHA256SUMS`](media/organ-vessel-body-link-20260913/SHA256SUMS).

Reproduce it with:

```sh
PYENV_VERSION=3.10.12 PYTHONPATH=src python -m numilab_human.vessel_body_links \
  --registration Docs/media/organ-vessel-registration-20260913/registration.json \
  --human-manifest Docs/media/organ-vessel-body-link-20260913/authoritative/myosim-fullbody-reference.manifest.json \
  --output Docs/media/organ-vessel-body-link-20260913/body-links.json
```

This closes exact source/core body-frame bookkeeping for the six surfaces. It
does not make a surface a vessel tube or lumen, and it does not assign a
mechanical blood mass, pressure-gradient reaction, tissue exchange, calibrated
wall material or density, subject calibration, or standing/walking behavior.
Those gates remain explicit in the receipt.

The focused regression is `tests/test_vessel_registration.py` (`7 passed`;
`68.35 s`). The same hashes and NHRIGID2 header were checked on `ssh
macmini`; the retained output is
[`macmini-validation.txt`](media/organ-vessel-body-link-20260913/macmini-validation.txt).
