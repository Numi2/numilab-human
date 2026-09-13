# Torso organ and nerve body-link receipt, 13 September 2026

The source-bound torso visual payload is now joined to the exact MyoSim
source/core body catalog through a strict receipt. It covers the five selected
organ surfaces (heart, stomach, pancreas, both kidneys), six named systemic
vessel surfaces, and the spinal-cord surface. Every row retains its
BodyParts3D member hash, source concept/name, layer, named MyoSim body,
source-body ID, core-body index, and default world pose from the hash-bound
`NHRIGID2` manifest.

The receipt consumes the pinned BodyParts3D relation tables, the native
`bodyparts3d-myosim-torso-anatomy.manifest.json`, the physical-M4 visual
receipt, and the authoritative MyoSim manifest. Its output is
[`body-links.json`](media/organ-torso-body-link-20260913/body-links.json),
with receipt SHA-256
`f7079b2dce816ef2d8b04a05c2acf34042f71085ef2e26700be929babc563bbe`. The
remote hash-and-schema check passed on `ssh macmini` (Apple M4 Pro); the
command output and file hashes are retained in
[`media/organ-torso-body-link-20260913/`](media/organ-torso-body-link-20260913/).

This closes source membership, visual source-to-world registration, and named
source/core body-frame bookkeeping for these twelve surfaces. It does not
create organ FEM/MPM, vessel tube or lumen mechanics, neural mechanics,
mechanical mass, materials or calibrated density, pressure-gradient reaction,
blood/tissue exchange, anatomical loading, subject calibration, standing,
recovery, or walking. The existing vessel and cardiac-cavity receipts remain
independent owners for their narrower bridge checks.

Reproduce it with:

```sh
numi human-organ-torso-body-links \
  --output Docs/media/organ-torso-body-link-20260913/body-links.json
```

The focused regression is `tests/test_torso_body_links.py`; it checks all
three source/body/payload identities and rejects map or payload tampering.
