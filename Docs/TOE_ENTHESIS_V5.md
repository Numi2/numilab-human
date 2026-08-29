# Numi Human toe enthesis v5

## Outcome

This first investigation corrected a real lesser-toe presentation ambiguity,
but it did not identify the defect the user reported: that defect was the
hallux. BodyParts3D contains five correctly ordered toe chains on each side;
the largest adjacent source-mesh gap is below 0.8 mm. The biomechanical source
provides one EDL and one FDL route per side for four anatomical lesser-toe
slips. Treating that representative route as one digit had made its terminal
surface read one toe off.

The v5 compiler now preserves the one source route and force law while adding
an explicit semantic enthesis map:

- left EDL/FDL: `FJ3179`, `FJ3180`, `FJ3181`, `FJ3185`, the distal phalanges
  of toes 2 through 5;
- right EDL/FDL: `FJ3189`, `FJ3190`, `FJ3191`, `FJ3195`;
- left EHL/FHL: hallux distal phalanx `FJ3182` only;
- right EHL/FHL: hallux distal phalanx `FJ3192` only.

For a lumped EDL/FDL endpoint, `NHTENDON2` selects one exact point on each of
the four named lesser-toe surfaces. Four precompiled 3x3 maps distribute the
single terminal force while conserving its resultant and its moment about the
unchanged source point. The maximum semantic span is 34.24 mm, below the 40 mm
toe-specific limit; sampled force amplification is 1.05--2.29. Hallux routes
use the existing connected single-bone envelope.

This admits eight bilateral toe insertions, raising whole-body distributed
coverage from 296 to 304 endpoints. The other 528 endpoints retain their exact
source-point fallback. Endpoint migration remains zero.

## Visual binding

The corresponding BodyParts3D EDL, EHL, FDL, and FHL terminal surfaces now
enter the exact toes-body frame as complete distal bands. An 8 mm exact
source-triangle lock and an 18 mm feather are combined with a smooth terminal
longitudinal band, avoiding the isolated weighted vertices that produced an
angular extra-slip silhouette. On the left, the combined locked/feathered
counts are 876/664 for EDL, 213/73 for EHL, 628/425 for FDL, and 245/114 for
FHL. This changes visual kinematic ownership only; the mechanical change is
the separate source-point-preserving `NHTENDON2` envelope.

These v7 counts are retained as the historical lesser-toe record. The later
[hallux v8 correction](HALLUX_ENTHESIS_V8.md) filters disconnected EHL/FHL
source shards and closes the EHL display seam against the exact big-toe distal
phalanx without changing v5 mechanics.

## Apple M4 Pro qualification

Runtime code `45fede450ba889b8feb1df0a8330db3c31706497` decoded and executed the
v4 bone, v7 tissue, and v5 tendon payloads with SHA-256 values:

- `969974058f5121bd0ef35689bbdb78b6aa2caba31920fa52193e218ad130efd6`;
- `d3f6f3501c2a48a42677cfe940d7d7001e912cc4c8ea7979d717f6a61aabfa8b`;
- `d563b10db8d27fdbed15d8eb196f8a57c6e6844126f91944b338917582f0aa97`.

The Metal reference probe transferred all 832 endpoints with 304 envelopes,
CPU/Metal nodal-force parity, zero single-scatter generalized-force difference,
and byte-identical replay. The canonical 64 assisted plus 64
assistance-removed 100 microsecond stand transaction then produced 106,496
accepted transfers: 38,912 four-node envelopes and 67,584 source-point
fallbacks. Maximum force/moment residuals were `1.72633488546e-4 N` and
`2.44306352215e-6 N m`; replay was bitwise.

The [clean left-foot views](media/numi-human-toe-enthesis-v5-2048/clean/),
[isolated EDL views](media/numi-human-toe-enthesis-v5-2048/edl/), and
[isolated FDL views](media/numi-human-toe-enthesis-v5-2048/fdl/) were rendered
at 2048 px from front, oblique, side, and rear on the same Apple M4 Pro. The
[qualification transcript](media/numi-human-toe-enthesis-v5-2048/qualification.transcript.txt),
[payload manifests](media/numi-human-toe-enthesis-v5-2048/manifests/), and
[checksums](media/numi-human-toe-enthesis-v5-2048/checksums.sha256) retain the
device and source identities.

## Boundary

MyoSim still has one articulated `toes` rigid body per side. This correction
does not invent four independent toe actuators or interphalangeal dynamics.
The four lesser-toe nodes are an inferred, source-registered distribution of
one lumped route wrench, not source-authored enthesis measurements, a
deformable tendon continuum, clinical anatomy, stable standing, or gait
validation.
