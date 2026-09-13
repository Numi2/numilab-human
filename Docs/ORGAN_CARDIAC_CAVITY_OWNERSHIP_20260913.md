# Cardiac cavity ownership candidates, 13 September 2026

The exact source-preserving cardiac partition now has a dedicated integration
receipt. It binds both right-heart ownership candidates (right-atrium priority
and right-ventricle priority) to the four-cavity bridge and the torso
source/core body link. Both emitted candidate interiors are disjoint, the
source union and source-exclusive regions are preserved, and the shared source
volume is retained from the partition artifact.

The receipt is [`ownership.json`](media/organ-cardiac-cavity-ownership-20260913/ownership.json)
with SHA-256
`1491d844567dc555ddb48a03cc3ccc60052b64173629fb0e1504b5d6187b11f9`. The
hash/schema check passed locally and on the physical M4 Pro through `ssh
macmini`; logs and hashes are retained in
[`media/organ-cardiac-cavity-ownership-20260913/`](media/organ-cardiac-cavity-ownership-20260913/).

No biological atrioventricular interface is selected. The original source
cavity domains remain overlapping, so this receipt does not assign a physical
cavity volume, density, blood mass, pressure reaction, tissue exchange,
physiological calibration, standing, or walking. It is the handoff required
before a future anatomical interface and conservative mass owner can be
selected.

Reproduce it with:

```sh
numi human-organ-cardiac-cavity-ownership \
  --output Docs/media/organ-cardiac-cavity-ownership-20260913/ownership.json
```

The focused regression is `tests/test_cardiac_cavity_ownership.py`; it rejects
pre-selected candidates and bridge geometry identity tampering.
