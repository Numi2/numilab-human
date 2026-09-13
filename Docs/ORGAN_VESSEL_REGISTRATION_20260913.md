# Source vessel frame registration, 13 September 2026

The new `organ-vessel-registration` owner binds the six named systemic vessel
surfaces in `bodyparts3d-myosim-torso-anatomy-map.v1.json` to the pinned
BodyParts3D 4.0 source archive, the physiology-template regions, and the
provisional source-to-MyoSim world transform.  It recomputes the source
membership and the current organ-moment receipt before emitting the immutable
[registration receipt](media/organ-vessel-registration-20260913/registration.json).

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

Reproduce it with:

```sh
PYTHONPATH=src python3 -m numilab_human.cli organ-vessel-registration \
  --output Docs/media/organ-vessel-registration-20260913/registration.json
```

The focused regression is `tests/test_vessel_registration.py`; it checks all
six source hashes, deterministic output, immutable writes, archive provenance,
and fail-closed transform tampering.
