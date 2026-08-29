# Numi Human hallux enthesis v8

## Outcome

The reported defect is the big toe, not a one-digit shift of the lesser-toe
chains. BodyParts3D and the mechanical source agree on the bilateral hallux
identity:

- left EHL/FHL insertion: distal phalanx `FJ3182` on MyoSim body `toes_l`;
- right EHL/FHL insertion: distal phalanx `FJ3192` on MyoSim body `toes_r`.

`NHTENDON2` v5 already preserves the exact EHL/FHL source endpoints and
distributes each terminal force over a four-node connected patch of that one
named distal phalanx. The v8 change does not alter those mechanics. It repairs
the corresponding BodyParts3D visual registration.

## Visual correction

The four BodyParts3D hallucis OBJ members contain one complete anatomical sheet
plus disconnected export shards. Sparse tibia/calcaneus/toes posing can turn
those shards into false terminal fragments. V8 retains the exact dominant
edge-connected sheet without filling, welding, or remeshing:

| Source member | Anatomy | Retained triangles | Discarded components |
| --- | --- | ---: | ---: |
| `FJ1408` | right EHL | 1,480 / 1,668 | 85 |
| `FJ1408M` | left EHL | 1,481 / 1,668 | 83 |
| `FJ1415` | right FHL | 2,677 / 2,758 | 45 |
| `FJ1415M` | left FHL | 2,677 / 2,758 | 45 |

The source EHL sheets also stop 7.4005/7.4016 mm from the right/left distal
hallux surfaces. For display only, 73 source-proximate vertices per side are
feathered onto the exact named bone triangles with a 0.35 mm interior inset.
Maximum corrections are 8.38/8.41 mm. The FHL sheets already contact the named
distal phalanges within 2 micrometres and are not projected.

This visual correction changes neither a MyoSim site nor an endpoint force.
The manifest records the source gap, retained topology, projection, and zero
source-endpoint migration.

## Apple M4 Pro validation

Runtime code `45fede450ba889b8feb1df0a8330db3c31706497` decoded and rendered the
v4 bone, v8 tissue, and unchanged v5 tendon payloads on Apple M4 Pro / Metal 4.
Their SHA-256 values are:

- bone: `969974058f5121bd0ef35689bbdb78b6aa2caba31920fa52193e218ad130efd6`;
- tissue: `e04e9d88c87e66a573ef677c9c3c93f67dc481e5dadcfbcb9a0c9b5630a0007a`;
- tendon: `d563b10db8d27fdbed15d8eb196f8a57c6e6844126f91944b338917582f0aa97`.

The [left clean views](media/numi-human-hallux-enthesis-v8-2048/clean-left/)
and [right clean views](media/numi-human-hallux-enthesis-v8-2048/clean-right/)
were reviewed from front, oblique, side, and rear at 2048 px with eight temporal
and eight area-light samples. The isolated [left EHL](media/numi-human-hallux-enthesis-v8-2048/ehl-left/)
and [left FHL](media/numi-human-hallux-enthesis-v8-2048/fhl-left/) views expose
the cyan source routes and actual tan four-node envelopes against the corrected
red source surfaces. Mirrored right EHL/FHL diagnostics were also inspected.

Bounded 0.2-activation EHL and FHL checks each evaluated all 416 source-force
records in two Apple Metal transactions before the FP64 pose update. They are
activation/posing checks, not persistent standing evidence; the unchanged v5
persistent transaction remains the force-transfer qualification. The retained
[transcripts](media/numi-human-hallux-enthesis-v8-2048/active/),
[manifests](media/numi-human-hallux-enthesis-v8-2048/manifests/), and
[checksums](media/numi-human-hallux-enthesis-v8-2048/checksums.sha256) make the
device and inputs inspectable.

## Boundary

MyoSim still exposes one `toes` rigid body per side. EHL/FHL are active
muscle-tendon routes with bone-surface force transfer, but this build does not
have an independently articulated big-toe MTP/IP chain, a deformable tendon
continuum, calibrated enthesis stress, stable standing, gait validation, or a
photorealistic exterior.
