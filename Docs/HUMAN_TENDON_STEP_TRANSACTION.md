# NumiLab Human per-step tendon-load transaction

## Outcome

`NHTENDON2` now runs inside every accepted step of the persistent Apple-native
Human horizon. Current-pose MyoSim route force remains the rigid-body authority;
the tendon pass converts each of the 832 origin/insertion terminal forces into
either its four-node BodyParts3D bone envelope or its explicit source-point
fallback. The stand kernel validates the complete environment-major transfer
before advancing `q` or `v`.

The distributed generalized correction remains a wrench-equivalence diagnostic.
It is never added to the MyoSim generalized force, so the new path cannot
double-count muscle force or turn tendon attachment into direct joint torque.

## Device composition boundary

After each stand encoder, an optional borrowed callback receives the same
command buffer plus immutable bindings/envelopes, current body poses, exact
terminal/nodal force records, generalized-correction diagnostics, and stand
status. Because the full horizon is encoded before execution, a physical
consumer must gate writes on success with the expected completed-step count.
A bone FEM/MPM consumer can reconstruct each envelope node from
its body-local coordinate and body pose, then encode its own load assembly
without a host readback. The callback may encode only: it cannot commit, wait,
retain, or replace any borrowed Metal object. If encoding rejects, its abort
hook runs and the enclosing result is not published.

The retained qualification uses a real blit consumer in that callback. It
copies the accepted transfer, correction, and status buffers on the same
command buffer; the snapshots are byte-identical to final host publication.
This proves the composition boundary and exact load visibility. It does not
claim that a deformable bone or tendon continuum has already consumed the
loads.

## Apple M4 Pro qualification

The Mac mini ran two eight-step horizons at `100 us`: an assisted phase and a
zero-root-wrench phase. All 416 MyoSim routes were reevaluated every step.

- 16 accepted persistent steps;
- 13,312 endpoint load transactions;
- 4,720 four-node BodyParts3D envelope transfers;
- 8,592 explicit source-point fallbacks;
- zero transfer failures;
- maximum force residual `1.079e-5 N`;
- maximum source-point moment residual `2.615e-7 Nm`;
- maximum absolute generalized correction `6.104e-5`;
- tendon-enabled and tendon-disabled one-step `q`/`v` were bitwise identical;
- a rejecting borrowed consumer left the caller result unchanged and invoked
  its abort hook exactly once;
- the assisted-plus-unassisted replay, including final tendon loads and
  diagnostics, was byte-identical.

The broader standalone 832-endpoint CPU/Metal parity probe also remained
unchanged: maximum nodal-force disagreement was `1.069e-4 N`, and its internal
Metal replay remained byte-identical.

## Four-angle visual inspection

The retained 2048 px front, oblique, side, and rear frames show the right
anconeus source surface, its unchanged MyoSim route, both endpoint envelopes,
and the owning humerus/ulna after the persistent transaction. Envelope coverage
was 182 / 446 / 939 / 1,160 pixels. Direct inspection confirmed that the route
and its warm four-node terminal fans stay on the named bone surfaces in every
view; no endpoint floats away from the skeleton.

The [capture transcript](media/numi-human-tendon-step-transaction-v3-2048/anconeus/capture.transcript.txt)
retains the device, counters, parity, rollback, replay, contact, and boundary
records. The [checksums](media/numi-human-tendon-step-transaction-v3-2048/checksums.sha256)
cover all four PNGs plus the native visual pack and metadata.

These are exposed mechanical-anatomy diagnostics, not a realistic skin/fat/
fascia render. The attachment coordinates remain simulation-inferred rather
than measured entheses, and the current stand remains a bounded low-velocity
horizon without exact high-speed RNEA/Jdot-v, calibrated passive equilibrium,
or a deformable tendon/bone material solve.
