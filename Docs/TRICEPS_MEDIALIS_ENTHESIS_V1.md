# Bilateral triceps medialis enthesis v1

The NHTENDON3 compiler now resolves the final upper-limb point fallback: the
left triceps medialis (`TRImed_l`) origin on the exact named BodyParts3D left
humerus (`FJ3262`). The pinned MyoSim left and right source sites are exact
local-frame reflections, but the independently sampled BodyParts3D humeri are
not assumed to be identical.

The compiler therefore uses the already admitted right `TRImed` four-node
patch only as a homologous seed. It reflects that seed, projects every node
onto an exact triangle of the left humerus, recomputes the wrench-equivalent
weights, and applies the unchanged distance, radius, amplification,
force-closure, and moment-closure gates. The MyoSim endpoint does not move.
Any source-site mismatch or failed target-surface projection is rejected.

## Compiled result

The new NHTENDON3 payload contains 642 distributed envelopes and 190 explicit
source-point fallbacks, compared with 641 and 191 before this change. Its 832
mechanical endpoints remain complete. The left origin passes with:

- 0 endpoint migration;
- 6.377 mm source-to-surface distance;
- 11.998 mm patch radius;
- 3.452 sampled total-force amplification;
- `1.61e-15` force-closure residual;
- `1.25e-17 m` moment-closure residual;
- 0.359 mm maximum projection from the reflected seed to the exact left
  humerus surface.

The right homolog remains independently compiled on `FJ3368`; both insertions
remain exact four-node ulna envelopes. All 107 importer tests pass (six source-
environment tests skipped). The machine-readable compile receipt is
[compile-receipt.json](media/numi-human-triceps-medialis-enthesis-v1/compile-receipt.json).

## Runtime validation

The companion Metal runtime certificate executes bilateral muscles 227 and
290 against the generated payload. Apple M4 Pro one-step and eight-step runs
verify positive tendon tension, elbow torque and state response, distributed
force/moment closure, bilateral force parity, same-command-buffer consumption,
bitwise replay, and rejected-transaction rollback. The old 641-envelope
payload fails closed because `TRImed_l` is still a point binding.

## Evidence boundary

This is an anatomically named, zero-migration, distributed enthesis transfer
for one bilateral source route. It is not a volumetric or deformable tendon,
passive elbow capsule or ligament mechanics, articular contact, sustained
loaded elbow motion, subject calibration, or clinical validation.
