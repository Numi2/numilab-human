# Third-party source notices

No raw BodyParts3D or OpenSim source data is committed to this repository. The
Apache-2.0 MyoSim visual-progress frames and the CC-BY-4.0 BodyParts3D visual
derivatives listed below are tracked source-derived media; the importer records
every other source in local generated artifacts.

## BodyParts3D 4.0

- Upstream: <https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html>
- License: [CC BY 4.0](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)
- Required attribution: `BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International`.
- Imported material: 4.0 OBJ meshes, FMA identifiers, English labels, and both
  `is-a` and `part-of` hierarchy relationships.
- Tracked derivatives: the three `Docs/media/bodyparts3d-native-skin/` PNGs
  are unmodified-frame renders of the exact `FJ2810` full-skin source mesh.
  The current four PNGs plus native visual pack in
  `Docs/media/myosim-native-bodyparts-major-bones-27/` render 27 exact named
  BodyParts3D bone members under provisional MyoSim link transforms. Their
  source members, hashes, Core renderer revision, attribution, and
  non-registration boundary are recorded in `Docs/VISUAL_PROGRESS.md` and
  `Docs/VISUAL_VALIDATION.md`. The four PNGs plus native visual pack in
  `Docs/media/myosim-native-muscle-driven-major-bones-27/` use the same 27
  exact BodyParts3D source meshes after a bounded Core muscle-force state step;
  the source attribution and non-registration boundary remain unchanged. The
  earlier 18-mesh galleries remain tracked as provenance-preserving milestones.

## Z-Anatomy right-calf visual supplement

- Upstream: <https://github.com/Z-Anatomy/Models-of-human-anatomy>, revision
  `9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a`, `Startup.blend`.
- License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- Required attribution: `Z-Anatomy, Lluis Vinas, CC BY-SA 4.0`.
- Imported material: only the right lateral/medial gastrocnemius, soleus,
  calcaneal tendon, and a non-rendered right-calcaneus registration landmark.
  No other Z-Anatomy organs or ancillary assets are imported.
- Tracked derivatives: `Docs/media/myosim-native-zanatomy-calf-2048/` is a
  CC-BY-SA source-derived visual supplement. It keeps BodyParts3D articulated
  bones and copied named MyoSim body weights; it is not a replacement force or
  mechanics asset. The raw blend and intermediate Blender export remain local.

## OpenSim RajagopalLaiUhlrich2023

- Pinned source: `opensim-org/opensim-models` commit
  `d9b05d470b1a481c222372c85b75772faf8f7792`,
  `Models/Rajagopal/RajagopalLaiUhlrich2023.osim`.
- The file credits Rajagopal et al. (2016), Lai et al. (2017), and Uhlrich et
  al. (2022). The upstream model repository does not provide a repository-level
  software license. This repository therefore does **not** redistribute the
  `.osim` or generated values, and records the source URL, revision, file hash,
  and credits in local artifacts. Confirm intended distribution rights before
  publishing a derived mechanics package.

## MyoSim `myofullbody`

- Upstream: <https://github.com/MyoHub/myo_sim>, pinned commit
  `33c89c2bde282553dde3f526768eb3bdcfaa7649`.
- License: Apache License 2.0. The source archive and all generated local
  payloads retain that upstream notice.
- Imported material: full-body articulated segment definitions, source joint
  records, masses/inertias, spatial-tendon sites and sphere/cylinder wraps,
  and the authored MuJoCo `general` muscle parameters.
- The three tracked visual-progress PNGs are source-derived renders under the
  same Apache-2.0 terms; their exact checksums are recorded in
  `Docs/VISUAL_PROGRESS.md`.

## Mortensen 2018 cervical/hyoid model

- Upstream: <https://github.com/mjhmilla/kinematicPassengerModel>, pinned
  commit `b0eb96127ca07dea0266764e837faeaa397092b5`.
- License: MIT.
- Imported material: the `HYOID_Scaled` OpenSim 3 body-owned joint structure
  and 72 Millard muscle records. The model is preserved as an input to an
  explicit rest-pose registration; it is not redistributed here.

## MoBL-ARMS Upper Extremity Dynamic Model

- Upstream: <https://simtk.org/projects/upexdyn/>, bimanual OpenSim release
  `MobL_ARMS_OpenSim3_bimanual_model.zip` (file 6366).
- The official project terms say the model is open-sourced solely for
  **non-commercial** use, while also including BSD 3-Clause text and required
  publication acknowledgement. The explicit non-commercial condition governs
  this importer.
- The archive is not redistributed. SimTK requires an authenticated download;
  provide the original archive locally and acknowledge its terms to build.
- Required acknowledgement: Saul KR, Hu X, Goehler CM, Daly M, Vidt ME,
  Velisar A, Murray WM. *Benchmarking of dynamic simulation predictions in two
  software platforms using an upper limb musculoskeletal model.* CMBE 2015;
  18:1445-58.

### Public unimanual MoBL-ARMS 4.1 mirror

- Upstream mirror: <https://github.com/CEINMS-RT/UpperLimbModel>, pinned to
  revision `459e2ebbf47acb72b0ecbc59950a2d0a983d28db`, model
  `MOBL_ARMS_41.osim`.
- This is a public **unimanual** MoBL-ARMS 4.1 source variant, not a claim that
  it is the authenticated SimTK bimanual archive. It is available only through
  an explicit `--accept-upper-noncommercial-terms` opt-in and is not
  redistributed here.
- The model's own non-commercial MoBL terms and required acknowledgement above
  still apply; the repository-level Apache notice does not broaden the model's
  terms.

## Numi Lab boundary

The resulting manifest is a source-faithful import artifact, not a validated
medical model, real-time Numi simulation, material calibration, or clinical
claim.
