# Torso and axial source registration v1

## Outcome

The BodyParts3D rib cage, thoracic spine, and paired hip bones are now placed
from their corresponding pinned MyoSim source meshes instead of a shared torso
guess. Recompiling the paired `NHBONES1` and `NHTENDON3` artifacts closes all
56 ordinary single-bone distance failures while retaining the unchanged 12 mm
admission threshold. The resulting 832 endpoint laws contain 620 distributed
surface envelopes and 212 explicit source-site point laws. Eighteen of the 620
envelopes are the existing route-private migrated foot/hallux terminals.

This is a rest-geometry registration and force-transfer result. It adds no
joint, does not move an authored MyoSim endpoint, and does not claim a
deformable tendon, cartilage, disc, ligament, breathing, contact, standing, or
clinical model.

## Owning registration

The implementation uses three independently gated source correspondences:

- `thoracic_registration.py` fits each exact BodyParts3D T1-T12 member to the
  matching compiled MyoSim vertebral mesh with a proper-rigid transform. All
  28 named thoracic entheses pass, 20 point fallbacks become surface
  envelopes, and all 13 C7-T1, T1-T12, and T12-L1 continuity checks pass.
- `rib_registration.py` decomposes the pinned MyoSim rib-cage mesh into 36
  connected components and resolves exactly 24 rib components by topology,
  side, level, and rest position. It jointly gates bilateral order,
  costovertebral proximity, endpoint parity, and all 44 named rib entheses.
  Thirty-four point fallbacks become surface envelopes.
- `pelvis_registration.py` fits the exact left and right BodyParts3D hip bones
  to the corresponding MyoSim pelvis meshes. It gates bilateral endpoint
  parity, both sacroiliac transitions, and all 44 named hip endpoints. The two
  iliacus point fallbacks become surface envelopes.

Every transform changes only the source member's rest geometry inside its
existing mechanics body. The current mechanics articulation remains 157 Core
bodies and 51 joint equalities.

## Measured gates

| Region | Source members | Named endpoint gates | Prior envelopes retained | Point laws recovered | Maximum distance before | Maximum distance after | Continuity/parity witness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T1-T12 | 12 | 28/28 | 8 | 20 | 25.119 mm | 9.515 mm | maximum axial gap 4.690 mm |
| ribs | 24 | 44/44 | 10 | 34 | 31.280 mm | 11.856 mm | costovertebral gap 7.641 mm; bilateral gap parity 1.940 mm |
| hip bones | 2 | 44/44 | 42 | 2 | 14.736 mm | 11.152 mm | endpoint parity 1.915 mm; sacroiliac gaps 0.481/0.661 mm |

The rib fit retains documented atlas-shape allowances for left rib 5 and both
rib 12 held-out p90 values; their mean fits remain bounded. These are explicit
source-shape differences, not a global threshold relaxation or endpoint edit.

The promoted compiler artifacts replayed byte-identically:

| Artifact | SHA-256 |
| --- | --- |
| `NHBONES1` | `970037584995d365322a5859e1cdc0c471d23b271ccb2c04cf3d6bc2a1d41da7` |
| bone manifest | `6f6bb7ace465a54b202cb4e020219dbc50866530580115d474c743d0b91ce5ae` |
| `NHTENDON3` | `662786d26820d04a53c4d7b37ce495f08188add96e57c88286bdd691208e8c02` |
| tendon manifest | `34772620b4f46da5da90e84229f07541dab1d44b1c5f1d7419f9f08f98b05af5` |

## Apple M4 Pro execution

The exact paired `NHRIGID`/`NHMYO2`/`NHEQ1` reference artifacts and the new
bone/tendon artifacts passed the native reference probe on Apple M4 Pro. It
executed all 416 muscles, 832 endpoint transfers, 143 wraps, Metal
kinematics/Jacobians/generalized force, transactional tendon scatter, and
byte-identical tendon replay.

Measured maxima were 0.124346 N Metal muscle-force error against a
2866.659 N reference maximum, 0.0670443 generalized-force error against a
2153.3623 reference maximum, 0.000244141 N tendon force residual, and
0.000008126 N m tendon moment residual. The controller still reports
`balanced=false`; this is runtime and force-transfer evidence, not stable
standing.

The complete raw line is retained in
[`reference-probe.txt`](media/numi-human-torso-axial-source-registration-v1-2048/reference-probe.txt).

## Multi-angle visual review

The captures below were rendered at 2048 px on Apple M4 Pro and inspected from
front, oblique, side, and rear views. The clean views verify bilateral rib
ordering and the thoracic chain. The enthesis views verify surface envelopes
on the ribs/vertebrae and both iliacus origins on the hip bones.

<p align="center">
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/torso-clean/myosim-fullbody-articulated-bodyparts-bones-focus-body-20-front.png" width="24%" alt="Registered torso bones, front" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/torso-clean/myosim-fullbody-articulated-bodyparts-bones-focus-body-20-oblique.png" width="24%" alt="Registered torso bones, oblique" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/torso-clean/myosim-fullbody-articulated-bodyparts-bones-focus-body-20-side.png" width="24%" alt="Registered torso bones, side" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/torso-clean/myosim-fullbody-articulated-bodyparts-bones-focus-body-20-rear.png" width="24%" alt="Registered torso bones, rear" />
</p>

<p align="center">
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/torso-entheses/myosim-fullbody-articulated-bodyparts-bones-source-route-centrelines-tendon-attachment-envelopes-focus-body-20-front.png" width="24%" alt="Registered rib and vertebral entheses, front" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/torso-entheses/myosim-fullbody-articulated-bodyparts-bones-source-route-centrelines-tendon-attachment-envelopes-focus-body-20-oblique.png" width="24%" alt="Registered rib and vertebral entheses, oblique" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/torso-entheses/myosim-fullbody-articulated-bodyparts-bones-source-route-centrelines-tendon-attachment-envelopes-focus-body-20-side.png" width="24%" alt="Registered rib and vertebral entheses, side" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/torso-entheses/myosim-fullbody-articulated-bodyparts-bones-source-route-centrelines-tendon-attachment-envelopes-focus-body-20-rear.png" width="24%" alt="Registered rib and vertebral entheses, rear" />
</p>

<p align="center">
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/pelvis-entheses-correct/myosim-fullbody-articulated-bodyparts-bones-source-route-centrelines-tendon-attachment-envelopes-focus-body-128-front.png" width="24%" alt="Registered pelvis and iliacus entheses, front" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/pelvis-entheses-correct/myosim-fullbody-articulated-bodyparts-bones-source-route-centrelines-tendon-attachment-envelopes-focus-body-128-oblique.png" width="24%" alt="Registered pelvis and iliacus entheses, oblique" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/pelvis-entheses-correct/myosim-fullbody-articulated-bodyparts-bones-source-route-centrelines-tendon-attachment-envelopes-focus-body-128-side.png" width="24%" alt="Registered pelvis and iliacus entheses, side" />
  <img src="media/numi-human-torso-axial-source-registration-v1-2048/pelvis-entheses-correct/myosim-fullbody-articulated-bodyparts-bones-source-route-centrelines-tendon-attachment-envelopes-focus-body-128-rear.png" width="24%" alt="Registered pelvis and iliacus entheses, rear" />
</p>

The anterior rib ends do not touch the sternum because the selected
BodyParts3D layer contains bone and not the missing costal-cartilage bridge.
Pulling the ribs forward to hide this gap would make their vertebral and muscle
correspondence worse. Costal-cartilage geometry and compliant mechanics are the
correct next increment.

## Reproduction boundary

The three candidates are chained so each stage preserves all prior reviewed
registration data:

```sh
numi human myosim-thoracic-registration \
  --sources Sources \
  --registration Build/lower-limb-source-mesh-v2.registration.json \
  --source-audit Build/myosim-source-bone-proximity-v1.json \
  --tendon-manifest Build/lower-limb-source-mesh-v2-tendon/numi-human-tendon-attachments.manifest.json \
  --python <pinned-myosim-python> \
  --output Build/thoracic-source-mesh-v1.registration.json

numi human myosim-pelvis-registration \
  --sources Sources \
  --registration Build/thoracic-source-mesh-v1.registration.json \
  --source-audit Build/myosim-source-bone-proximity-v1.json \
  --tendon-manifest Build/thoracic-source-mesh-v1-tendon/numi-human-tendon-attachments.manifest.json \
  --python <pinned-myosim-python> \
  --output Build/axial-source-mesh-v1.registration.json

numi human myosim-rib-registration \
  --sources Sources \
  --registration Build/axial-source-mesh-v1.registration.json \
  --source-audit Build/myosim-source-bone-proximity-v1.json \
  --tendon-manifest Build/axial-source-mesh-v1-tendon/numi-human-tendon-attachments.manifest.json \
  --python <pinned-myosim-python> \
  --output Build/torso-axial-source-mesh-v1.registration.json
```

Promotion requires rebuilding paired bone and tendon artifacts, byte-identical
replay, the native M4 Pro probe, and review of all four visual angles. A
candidate JSON alone is not promotion evidence.
