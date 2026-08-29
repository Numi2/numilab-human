# Topology-aware exact-surface entheses v1

## Closure

The previous compiler could reject an otherwise close and correctly named
BodyParts3D surface when its local source mesh supplied fewer than four usable
vertices or the compass-quadrant heuristic selected an ill-conditioned
quartet. This was a force-transfer discretization problem, not evidence that
the authored MyoSim endpoint should move.

The fallback compiler now constructs a deterministic pool of at most 14
points from:

- exact barycentric points on the already selected nearest source triangle;
- real vertices inside that triangle's connected 12 mm geodesic neighborhood;
- deterministic directional extrema from that same neighborhood.

It evaluates at most 1,001 four-point combinations and admits only the
minimum-norm map that still satisfies every existing gate. It does not warp or
refine the BodyParts3D mesh, change a bone identity, move a MyoSim site, alter
a route, or add a second force law. Virtual points are stored with exact
triangle/barycentric provenance; the runtime continues to consume only four
positions and four `3 x 3` maps for the one terminal wrench.

Nearest-triangle ties now resolve to the lowest source triangle index within a
strict floating-point tolerance. This makes shared-edge decisions stable
across rebuilds while remaining on the same named bone surface.

## Compiler result

The exact v10 inputs were rebuilt with the unchanged 12 mm surface-distance
gate, 12 mm patch-radius gate, sampled total-force-amplification limit of 4.0,
`2e-6` unit-force residual limit, and `2e-8 m` unit-moment residual limit.

| Metric | v10 | v11 |
| --- | ---: | ---: |
| distributed surface envelopes | 282 | 364 |
| topology-aware exact-surface envelopes | 0 | 82 |
| exact source-point laws | 550 | 468 |
| surface coverage | 33.89% | 43.75% |
| endpoint migration | 0 | 0 |

The 82 recovered endpoints span 78 muscles and bodies 20, 41, 84, 91, 128,
131, 136, 145, and 150. They include two additional source-named thoracic
entheses, 40 additional declared hip/tibia/fibula entheses, and 40 direct
single-bone upper/lower-limb endpoints. All 282 prior envelopes remain
admitted on the same named bone. Six shared-edge triangle ties select a
deterministic equivalent triangle on that same surface; all 832 source points
remain exactly unchanged.

For the new 82 maps:

| Gate quantity | Measured range |
| --- | ---: |
| surface distance | `0.0037--11.9800 mm` |
| patch radius | `3.5009--11.9971 mm` |
| sampled total-force amplification | `1.00039--3.80959` |
| unit-force residual | `<= 3.725e-15` |
| unit-moment residual | `<= 4.652e-17 m` |

Nine close endpoints still fail conditioning after the topology-aware search:
right `CORB`, `BRA`, and `BRD` on the humerus; bilateral `addlong` insertions,
right `glmax2`, left `glmax3`, and bilateral `vasint` origins on the femora.
They remain explicit point laws. The other current fallback classes are
unchanged: 411 distance failures, 24 bodies without a registered bone surface,
20 unresolved multi-member endpoints, and four semantic toe-distance
failures.

The compiler was run twice from the same paired inputs. Both NHTENDON2 files
were byte-identical with SHA-256
`5b21a7e70aef7bf5b208f540cd6210c713bf287a31c15b43797265a3ed4d4bec`.

## Apple M4 Pro qualification

The paired artifacts were:

- coherent `NHBONES1` SHA-256
  `0efe0a20ba31cc838b4e76ad14fe89492227869256ae797551350ada339ef7ab`;
- `NHMYO2` SHA-256
  `9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05`;
- v11 `NHTENDON2` SHA-256
  `5b21a7e70aef7bf5b208f540cd6210c713bf287a31c15b43797265a3ed4d4bec`;
- Numi runtime revision
  `50cab6b69426ad28c268aa05738de71df9f88bf0`.

The 512 px persistent probe completed 16 assisted and 16
assistance-removed 100 us steps across all 416 routes. It executed 26,624
endpoint transfers: 11,648 distributed envelopes and 14,976 point fallbacks.
Maximum live float residuals were `1.9601e-4 N` and `1.8631e-6 N m`.
The consumer used the same command-buffer snapshot, rejection preserved state,
and replay was bitwise. The compiled standing transaction still reports
`balanced=false`; this is a load-path qualification, not stable stance.

## Four-angle diagnostic review

These views deliberately omit soft-tissue surfaces so the exact bone, cyan
source route, and warm four-node transfer patch can be inspected without
occlusion. They are mechanics diagnostics, not realism renders.

### Right shoulder and upper arm

<p align="center">
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-upper-arm-diagnostic-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-41-front.png" width="24%" alt="Topology-aware right upper-arm entheses, front" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-upper-arm-diagnostic-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-41-oblique.png" width="24%" alt="Topology-aware right upper-arm entheses, oblique" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-upper-arm-diagnostic-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-41-side.png" width="24%" alt="Topology-aware right upper-arm entheses, side occlusion check" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-upper-arm-diagnostic-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-41-rear.png" width="24%" alt="Topology-aware right upper-arm entheses, rear" />
</p>

Four selected routes (`DELT1`, `PECM1`, `LAT1`, and `TRIlat`) execute for 16
accepted steps. Six selected endpoint envelopes render. Front, oblique, and
rear expose patches on the humerus; the side camera is substantially
bone-occluded and is retained as a failed visibility angle rather than
silently discarded.

### Right femur and knee span

<p align="center">
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-femur-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-131-front.png" width="24%" alt="Topology-aware right femur entheses, front" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-femur-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-131-oblique.png" width="24%" alt="Topology-aware right femur entheses, oblique" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-femur-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-131-side.png" width="24%" alt="Topology-aware right femur entheses, side" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-femur-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-131-rear.png" width="24%" alt="Topology-aware right femur entheses, rear" />
</p>

Four selected routes (`addbrev`, `addmagMid`, `gasmed`, and `vaslat`) render
seven envelopes. Front, oblique, and rear expose proximal and distal femoral
patches. The side view verifies depth but hides most warm pixels behind the
femur.

### Right shank

<p align="center">
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-shank-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-136-front.png" width="24%" alt="Topology-aware right shank entheses, front" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-shank-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-136-oblique.png" width="24%" alt="Topology-aware right shank entheses, oblique occlusion check" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-shank-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-136-side.png" width="24%" alt="Topology-aware right shank entheses, side" />
  <img src="media/numi-human-topology-aware-entheses-v1-2048/right-shank-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-136-rear.png" width="24%" alt="Topology-aware right shank entheses, rear" />
</p>

Five selected routes (`edl`, `ehl`, `fhl`, `perbrev`, and `tibant`) render
five envelopes. Front, side, and rear expose tibial/fibular patch pixels; the
oblique camera reports zero visible envelope pixels because the patches are
occluded. The cyan routes also expose opposite endpoints that remain point
fallbacks. Those visible gaps belong to the registration/semantic backlog and
are not claimed as repaired by this increment.

The [whole-body transcript](media/numi-human-topology-aware-entheses-v1-2048/stand-smoke-512.transcript.txt),
three focused transcripts, exact manifests, visual JSON receipts, and
[checksums](media/numi-human-topology-aware-entheses-v1-2048/checksums.sha256)
are retained with the frames. The larger render packs remain on the Mac mini
under `qualification-artifacts/topology-aware-entheses-v11-paired` rather than
duplicating 75 MB in Git.

## Evidence boundary and next gate

This increment proves a deterministic exact-surface quadrature and its live
single-law force transfer. It is not a deformable tendon continuum,
fibrocartilage model, bone-stress solve, clinical enthesis map, or proof that
every cyan route visually terminates on anatomy. The
[source-bone proximity audit](SOURCE_BONE_PROXIMITY_V1.md) now separates the
distance backlog into 256 genuine BodyParts3D registration candidates and 159
sites that are already non-bone-adjacent in the mechanics source. The first
bounded target is the 176 bilateral upper-limb and hand candidates. A coherent
regularized registration may propose that correspondence, but semantic bone
ownership and the unchanged mechanical gates must still admit each endpoint
separately.

For that next gate, [Coherent Point Drift](https://arxiv.org/abs/0905.2635)
is a useful primary reference for a coherently regularized non-rigid
point-set proposal, while the classic
[free-form deformation registration](https://webdocs.cs.ualberta.ca/~vis/readingMedIm/papers/RueckertFreeForm.pdf)
provides an affine-plus-local B-spline alternative. Neither method supplies
anatomical endpoint semantics by itself. NumiLab should use the deformation
only to propose a source-to-source receipt, then require named bone ownership,
held-out landmarks, invertibility/fold checks, and the existing per-endpoint
distance/wrench gates before promotion.
