# Cardiac cavity body-frame links — 13 September 2026

The `cardiac_body_links` owner connects the four exact BodyParts3D cavity
members already used by the CVSim bridge to the hash-bound MyoSim `torso`
source/core body (source body `9`, core body `20`). The immutable
[receipt](media/organ-cardiac-cavity-body-link-20260913/body-links.json) covers
right atrium `FJ2424`, right ventricle `FJ2423`, left atrium `FJ2425`, and left
ventricle `FJ2422`.

The receipt SHA-256 is
`00050998a578e4f6853388fc3b90c529ac740b636298c971755648ccf4f837c9`; the
four-entry map SHA-256 is
`5f85a7a13c661e0ab093d609128597624ee553b89642735d3838c3340d2c95fa`.

The owner validates the BodyParts3D `part_of` relations, the existing bridge
identity and source-member hashes, the complete 103-source-body NHRIGID2
mapping, and the packed torso pose. It leaves the bridge's intersecting cavity
domains unresolved and does not assign a physical cavity volume, blood mass,
wall material or density, pressure reaction, tissue exchange, calibration, or
behavior.

Reproduce it with:

```sh
PYENV_VERSION=3.10.12 PYTHONPATH=src python -m numilab_human.cardiac_body_links \
  --output Docs/media/organ-cardiac-cavity-body-link-20260913/body-links.json
```

The focused regression is `tests/test_cardiac_body_links.py` (`3 passed`). The
same receipt and NHRIGID2 hashes pass on `ssh macmini`; retained output is in
[`macmini-validation.txt`](media/organ-cardiac-cavity-body-link-20260913/macmini-validation.txt).
The artifact directory is covered by
[`SHA256SUMS`](media/organ-cardiac-cavity-body-link-20260913/SHA256SUMS).
