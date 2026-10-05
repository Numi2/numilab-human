# Named major vessels in the resting scene

The existing NHANAT5 payload now includes 61 additional BodyParts3D v4 source
surfaces from 54 explicitly sided source concepts: carotid, jugular,
subclavian, axillary, brachial, radial, ulnar, iliac, femoral, popliteal, tibial
and renal vessels. The existing source compiler registers each surface to its
declared MyoSim body frame. The existing CVSim21 regional states retain sole
ownership of hydraulic volumes, pressures and flows; no vessel mass or wall
mechanics is added. Distal microvessels remain reduced regional compartments.

All preparation, audits, native compilation and execution were performed on
the SSH Mac mini. The independent source-face parser and coordinate round-trip
check in `major-vessels-source-audit-001.json` verifies all 61 surfaces. The
maximum registered coordinate error is 1.490153145e-8 m. The previous records,
vertices and indices are byte-preserved. This establishes source preservation,
not loaded interface or vessel-wall qualification.

The assembled 524-surface payload has 1,258,946 vertices and 6,001,692 indices,
SHA-256 `9ec1c6cc617b0fba86f45eb005472e8ca8a7f984d2797b8710a55d5ecbe593e6`.
It was admitted by the native viewer, which completed 500 accepted 2 ms steps
in 13.527701292 native wall seconds (RTF 0.07392). `invocation.json` and
`run-metadata.json` retain exact loaded files and hashes. The unretimed recording
has 17 actual frames, and the seven inspection layers were extracted at exact
presentation timestamps. `frame-vessels.png` is the actual vessels frame.

This is a one-second anatomical loading/presentation check. Known respiratory
and cardiac dynamic geometry defects remain; it is neither endurance nor
anatomical qualification. The Mac mini was at its login window, so the movie
captures the real Metal renderer and native AppKit panel, while interactive
desktop presentation remains unverified. Full output and recording:
`/Users/n/numi-human-resting-evidence-20261005/native-vessels-admission-002`.

The failed first admission is retained in `native-vessels-admission-001`:
appending geometry after the cardiac derivation invalidated its whole-payload
hash, and the loader rejected it before stepping. The corrected order is
passive viscera → major vessels → cardiac partition/binding. A separate
authoring attempt with the wrong `--numilab-source` directory also failed
before output and remains in `major-vessels-cardiac-admission-002.log`.
The successful authoring log is `major-vessels-cardiac-admission-003.log`.
Exact cardiac auditing was deliberately skipped for this presentation check;
the prior dynamic-geometry failures are not overridden by its successful load.

Preparation uses `python -m numilab_human.resting_major_vessels --base-receipt
RECEIPT --output NEW_DIRECTORY`, followed by the existing cardiac authoring
tool. The admitted receipt is at
`/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/major-vessels-cardiac-admission-003/resting-anatomy-receipt.json`.

BodyParts3D, © The Database Center for Life Science licensed under CC
Attribution 4.0 International. These are reference atlas sources registered to
MyoSim, not measurements of a single individual. Existing Z-Anatomy CC-BY-SA
4.0 and MyoSim Apache-2.0 provenance is retained by the composite receipt.
