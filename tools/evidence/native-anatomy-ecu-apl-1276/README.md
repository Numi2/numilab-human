# Four passive muscle repairs in the integrated native human

The bilateral extensor carpi ulnaris and abductor pollicis longus surfaces now
run without their previous self-crossings in the existing resting human scene.
These are stable anatomical rows 115, 116, 147 and 148, not replacement organs or
a separate simulation. The 146 other surface rows and every binding table are
byte-preserved; physical mass, muscle routes, tendons, contact, cardiac,
respiratory and Brain owners are unchanged.

Run on the retained SSH Mac mini installation:

```sh
ssh macmini '/usr/bin/python3 /Users/n/numi-human-anatomy-completion-1276/tools/evidence/native-anatomy-ecu-apl-1276/reproduce.py'
```

Use --prepare-only to verify retained inputs and create a fresh launch declaration
without starting Metal. Every reproduction uses a fresh output directory, the
existing owner launcher, exact input checks and the shared native-owner lock.
This requires the retained Mini runtime/assets; it is not a portable installer.

## Demonstrated

- Existing topology-preserving edge-collapse owner: five source self-pairs per
  ECU side and four per APL side reduced to zero. Maximum displacement from
  original source ancestry is 0.128748 mm and 0.204398 mm respectively.
- Closed topology, component count and Euler characteristic preserved. The
  APL's pure-binding endpoint proxies are unchanged; ECU has no pure-binding
  surface proxies. These are not measured attachment footprints.
- All 16 retained baseline/intervention poses: full candidate self-count zero;
  changed-region exact scans against 859 other captured surfaces add zero
  intersection pairs. This checks the edits, not every pre-existing interface.
- Actual native run: 8,000 accepted steps at Float32 0.002 seconds, nominally
  16 seconds. All four surfaces are closed and self-free in all eight actual
  captured geometries, including the sampled breathing cycle after 10 seconds.
- 1,000 synchronized coupled-state rows. Respiratory, cardiac, gas, activation
  and normal-impulse columns match the unchanged 310-second baseline's prefix.
  The complete CSV is **not** identical: this run enables a quaternion audit
  earlier, after which projection diagnostics and contact gaps differ.
  Maximum gap difference is 0.545 micrometres. Association with that setting
  is not proof of cause; matched-setting mechanical comparison remains open.
- Native framebuffer recording: 252 image frames, five timing markers,
  monotonic presentation, 160.162 seconds recording duration, maximum gap
  2.480 seconds. Seven anatomical layers are represented. It is a continuous
  native-renderer recording, not independent evidence of desktop visibility.

Owner wall time was 164.006 seconds on the M4 Pro Mini (nominal real-time factor
0.09756). Performance work is paused; this run was for anatomy verification.

## Identity and retained evidence

Human source: cde05f22cda41600ccbfcbb04a82842e9d6248b4.
Native Lab runtime source: f47429c30289fef2de2cade0962a2e2d8d91fe44.
Brain source: a1cf7218fae5d26f9aed9d845047fc6c472ad596.

Parent tissue SHA256:
6579d06d358bfcf51f964feb57425fe568c467fd361930bce46d376ec8aedaa4.
Composed tissue SHA256:
3cb4e877969ea0c651829838a942fc68e6dd401bce2d341fdcd7f1fa7d5845aa.

The exact native command, 37 immutable inputs, environment, execution receipt,
geometry patches, source derivation, pose/interface audits, compact traces and
movie are retained here. Large source assets and eight MRVPACK captures remain
on the Mini at
/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/ecu-apl-native-composition-005.
retained-files.json binds each published copy to its retained source and hash.
Readable analysis scripts omit trailing blank lines; adjacent .py.gz files preserve
all exact executed bytes. retained-only-files.json records full binding and dense
contact traces kept on the Mini instead of duplicating them in Git.

The native preparation attempts 003 and 004 were rejected before simulation
because their requested capture IDs did not match the existing 32-step
presentation cadence. They remain retained. Attempt 005 uses valid cadence IDs.
Attempt 001 had an output-directory adapter error; 002 was preparation only.

## Scope and remaining work

This is an executable anatomy increment, **not completion of the full human
deliverable**. Other passive muscle intersections and the shoulder/rib and
ankle/calcaneus repairs remain under investigation. The unchanged whole body's
other tissues are not admitted by these four passing rows. A new five-minute
run of the final anatomy and matched respiratory intervention is still needed.

The mixed-source reference anatomy remains honestly identified as such.
BodyParts3D source provenance and its CC BY 4.0 attribution to DBCLS remain in
the original and composed manifests; MyoSim provenance remains attached to
the original muscle-route owner. New midpoint positions and canonical mean
binding weights are explicit numerical reconstructions, not measurements from
one person. Passive inspection surfaces do not become physical tissue-volume
owners through this repair. No clinical validation is claimed.
