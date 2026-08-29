# Upper-limb source-mesh registration v1

## Outcome

The bilateral humerus-to-finger registration is now promoted into the paired
`NHBONES1`/`NHTENDON2` validation path. Exact BodyParts3D geometry is rigidly
registered to the pinned compiled MyoSim bone meshes; MyoSim sites, muscle
paths, joints, masses, and force parameters are unchanged.

The final candidate covers 60 named bodies in 30 bilateral pairs. It uses
PCA-seeded, 90%-trimmed symmetric rigid ICP with every fifth deterministic
surface sample reserved before fitting. Bilateral mirror consistency, endpoint
distance, prior-envelope preservation, and default-pose continuity are hard
selection gates. The wrist correction translates each complete hand together;
it does not add a joint or independently articulate a finger.

| Gate | Result |
| --- | ---: |
| upper-limb/hand registration candidates | 166/166 within 12 mm |
| maximum candidate distance, before / after | 139.377 / 8.615 mm |
| previously admitted upper-limb envelopes preserved after continuity regularization | 45/45 |
| bilateral body pairs | 30 |
| default-pose shoulder/elbow/wrist/hand transitions | 52/52 pass |
| worst transition | 11.040 / 12.000 mm, bilateral ulna-to-triquetrum interval |
| maximum held-out surface p90 / outlier | 11.224 / 20.750 mm |
| maximum mirrored surface p90 / outlier | 9.020 / 20.219 mm |
| endpoint migration | 0 mm |

The held-out and mirrored outliers remain visible because cross-subject bone
surfaces are not identical. They are not used to weaken the endpoint gate.

The 12 mm ulnocarpal gate is not a direct ulna-to-triquetrum contact claim. The
pinned MyoSim bone-only source leaves 9.371 mm bilaterally, while the TFCC
includes an articular disc and ulnocarpal ligamentous structures in this
interval. The anatomy boundary is consistent with cadaveric TFCC studies of
the [ligamentous structure](https://pubmed.ncbi.nlm.nih.gov/9848546/) and
[carpal attachment](https://pubmed.ncbi.nlm.nih.gov/12358373/). No TFCC,
cartilage, ligament, or compliant-contact mechanics are implemented here.

## Direct visual review

Every frame below is a native 2048 px Apple M4 Pro render of the final paired
hash. White is exact BodyParts3D bone geometry, cyan is the exact current-pose
MyoSim route centreline, and the warm four-node fans are the executable
`NHTENDON2` transfer maps. The fans are force-transfer diagnostics, not a
fabricated tendon surface.

### Right hand and fingers

<p align="center">
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/right-hand-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-57-front.png" width="24%" alt="Registered right hand, front" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/right-hand-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-57-oblique.png" width="24%" alt="Registered right hand, oblique" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/right-hand-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-57-side.png" width="24%" alt="Registered right hand, side" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/right-hand-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-57-rear.png" width="24%" alt="Registered right hand, rear" />
</p>

### Left hand and fingers

<p align="center">
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/left-hand-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-107-front.png" width="24%" alt="Registered left hand, front" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/left-hand-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-107-oblique.png" width="24%" alt="Registered left hand, oblique" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/left-hand-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-107-side.png" width="24%" alt="Registered left hand, side" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/left-hand-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-107-rear.png" width="24%" alt="Registered left hand, rear" />
</p>

### Right elbow, forearm, and wrist

<p align="center">
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/right-elbow-wrist-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-43-front.png" width="24%" alt="Registered right elbow and wrist, front" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/right-elbow-wrist-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-43-oblique.png" width="24%" alt="Registered right elbow and wrist, oblique" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/right-elbow-wrist-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-43-side.png" width="24%" alt="Registered right elbow and wrist, side" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/right-elbow-wrist-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-43-rear.png" width="24%" alt="Registered right elbow and wrist, rear" />
</p>

### Left elbow, forearm, and wrist

<p align="center">
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/left-elbow-wrist-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-93-front.png" width="24%" alt="Registered left elbow and wrist, front" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/left-elbow-wrist-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-93-oblique.png" width="24%" alt="Registered left elbow and wrist, oblique" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/left-elbow-wrist-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-93-side.png" width="24%" alt="Registered left elbow and wrist, side" />
  <img src="media/numi-human-upper-limb-source-mesh-registration-v1-2048/left-elbow-wrist-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-93-rear.png" width="24%" alt="Registered left elbow and wrist, rear" />
</p>

Direct inspection found no swapped forearm bones, disconnected elbow or wrist
block, off-by-one digit assignment, individually shifted finger, or floating
terminal fan in the four selected views. Side-view occlusion can hide a route
or fan; the other cameras and pixel counters remain the cross-check.

## Tendon-law result

Recompiling the exact final bone/tendon pair produces 162 net distributed-law
gains and no loss relative to v11. Of the 166 intended registration targets,
159 become distributed envelopes; three additional same-body endpoints pass
the unchanged gates after the coherent geometry update.

| Disposition | v11 | final |
| --- | ---: | ---: |
| connected four-node surface envelope | 364 | 526 |
| exact source-site point law | 468 | 306 |
| surface coverage | 43.75% | 63.22% |

Seven intended targets remain exact point laws because their candidate
four-node patches fail conditioning: bilateral `TRImed` origins, left `FDP5`
insertion, and right `RI4`, `LU_RB4`, `UI_UB4`, and `RI5` origins. These are
not repaired by relaxing distance, force-amplification, or wrench gates.

The scapulae are also deliberately not rigidly promoted. Their ten candidates
cannot all pass the unchanged 12 mm endpoint gate: the best proper rigid fits
still leave 13.835 mm on the right and 16.197 mm on the left. The correct next
step is a source-landmark-constrained scapular correspondence using glenoid,
acromion, coracoid, and medial-border landmarks, or a superior compatible
source—not a global threshold increase.

## Native execution evidence

The final pair ran in revision `50cab6b69426ad28c268aa05738de71df9f88bf0`
of the `coupled` runtime checkout on Apple M4 Pro.

The reference probe evaluates 416 muscles and 832 terminal laws on Metal. It
reports 526 envelope transfers, 306 point transfers, zero endpoint migration,
`1.25885e-4 N` maximum Metal force residual, `4.30321e-6 N m` maximum moment
residual, and byte-identical tendon replay.

The full-body smoke executes 32 assisted plus 32 zero-root-wrench steps:
53,248 terminal transfers, including 33,664 envelope and 19,584 point
transfers. Maximum per-step residuals are `1.25827e-4 N` and
`1.86306e-6 N m`; deterministic replay is bitwise. This is a runtime and
force-transfer smoke, not a balance qualification: the compiled report still
says `balanced=false`.

The hand captures each apply a 0.05 increment to 31 selected source muscles
over the compiled posture while all 416 paths continue to run. The elbow/wrist
captures each select 17 routes. The exact transcripts and image hashes are in
the [media directory](media/numi-human-upper-limb-source-mesh-registration-v1-2048/)
and [checksum set](media/numi-human-upper-limb-source-mesh-registration-v1-2048/checksums.sha256).

## Reproduction and exact pair

The generated `Build/` products are reproducible and byte-identical across two
independent local runs:

| Artifact | SHA-256 |
| --- | --- |
| registration candidate | `e690400b88c02b192814b1e08bc9c7144eca270cafb77c502a5ab9327d4501ed` |
| `NHBONES1` | `3af416b81f4bc47c0552bc0f3babd6f3a8bfcb190ca172c16fe8b427d3f59035` |
| `NHTENDON2` | `f50e12d6f3c804822f857110d3ac526c476802541cb08a730e9c7a25d216b3f1` |

```sh
PYTHONPATH=src python3 -m numilab_human.cli myosim-upper-limb-registration \
  --sources Sources \
  --registration Build/coherent-body-v4.registration.json \
  --source-audit Build/myosim-source-bone-proximity-v1.json \
  --worklist Build/bodyparts-registration-worklist-v1.json \
  --tendon-manifest Build/numi-human-tendon-topology-v11/numi-human-tendon-attachments.manifest.json \
  --output Build/upper-limb-source-mesh-v1-final.registration.json \
  --python .venv-myosim/bin/python

PYTHONPATH=src python3 -m numilab_human.cli myosim-bodyparts-bone-payload \
  --sources Sources \
  --registration Build/upper-limb-source-mesh-v1-final.registration.json \
  --output Build/upper-limb-source-mesh-v1-final-bones

PYTHONPATH=src python3 -m numilab_human.cli numi-human-tendon-envelope-payload \
  --artifact Build/myosim-fullbody \
  --bone-artifact Build/upper-limb-source-mesh-v1-final-bones \
  --output Build/upper-limb-source-mesh-v1-final-tendon
```

## Evidence boundary

This increment proves deterministic source-mesh registration, exact paired
force-transfer compilation, native Apple execution, and inspected default-pose
continuity. It does not prove subject-specific or clinical registration,
photorealistic anatomy, deformable tendon/enthesis mechanics, TFCC/cartilage
contact, independent finger control, closed-loop balance, or realistic skin.
