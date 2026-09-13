# Source vessel frame registration, 13 September 2026

The new `organ-vessel-registration` owner binds the six named systemic vessel
surfaces in `bodyparts3d-myosim-torso-anatomy-map.v1.json` to the pinned
BodyParts3D 4.0 source archive, the physiology-template regions, and the
provisional source-to-MyoSim world transform.  It recomputes the source
membership and the current organ-moment receipt before emitting the immutable
[registration receipt](media/organ-vessel-registration-20260913/registration.json).

That first receipt is retained as historical evidence. A unit audit found that
its compiler applied the pinned millimetre-to-metre factor twice to moments that
were already authored in metres. The current source/world receipt is
[`media/organ-vessel-registration-corrected-20260913/registration.json`](media/organ-vessel-registration-corrected-20260913/registration.json),
which applies the declared uniform scale once; see
[`ORGAN_VESSEL_UNIT_CORRECTION_20260913.md`](ORGAN_VESSEL_UNIT_CORRECTION_20260913.md).
Native admission must use that corrected fixture.

The six bindings are ascending aorta (`FJ3413`), aortic arch (`FJ3411`),
descending aorta (`FJ3427`), abdominal aorta (`FJ1932`), superior vena cava
(`FJ3645`), and inferior vena cava (`FJ3441`).  Each row retains its source
member hash, source-frame centroid and central second moment, the target
MyoSim body name, and the transformed world-frame centroid and second moment.
The transform is checked as the pinned proper uniform scale and signed-axis
convention; source archive and registration hashes are carried into the
receipt identity.

This closes source-to-world frame bookkeeping for these six surfaces.  It does
not turn a surface into a vessel tube or lumen.  The receipt therefore keeps
body-link mechanics, centreline and cross-sectional area, wall material,
density, blood mass ownership, pressure-gradient reaction, tissue exchange,
subject calibration, and standing/walking explicitly false or unresolved.
The registration candidate remains visual-only and is not admitted to
collision or physics.

The follow-up [exact body-link receipt](ORGAN_VESSEL_BODY_LINK_20260913.md)
consumes the hash-bound MyoSim `NHRIGID2` manifest and payload and closes named
source/core body-frame bookkeeping for the six `myosim_body` fields. It still
does not admit body-link mechanics or any vessel physical owner.

Reproduce it with:

```sh
PYTHONPATH=src python3 -m numilab_human.cli organ-vessel-registration \
  --output Docs/media/organ-vessel-registration-20260913/registration.json
```

The focused regression is `tests/test_vessel_registration.py`; it checks all
six source hashes, deterministic output, immutable writes, archive provenance,
and fail-closed transform tampering.
