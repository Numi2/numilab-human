# Third-party source notices

No raw BodyParts3D, Z-Anatomy, or OpenSim source archives are committed to this
repository. The tracked media and generated scene inputs are source-derived;
source identities and asset fingerprints are recorded below or in the
corresponding local generated artifact.

## BodyParts3D 4.0

- Upstream: <https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html>
- License: [CC BY 4.0](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)
- Required attribution: `BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International`.
- Imported material: 4.0 OBJ meshes, FMA identifiers, English labels, and both
  `is-a` and `part-of` hierarchy relationships. The resting scene uses the direct
  4.0 archives isa_BP3D_4.0_obj_99.zip (SHA-256
  40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e) and
  partof_BP3D_4.0_obj_99.zip (SHA-256
  9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97).
  This direct 4.0 use is separate from the older BodyParts3D credit for
  material included within the Z-Anatomy source package below.
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
  The four terminal PNGs in `Docs/media/unassisted-standing-20260922/`
  render registered BodyParts3D bone and muscle surfaces at the accepted state
  after ten simulated seconds; their attribution, source/binary hashes and
  physical qualification limits are recorded in that directory's README.

## Z-Anatomy thorax, liver, diaphragm, and calf references

- Upstream: https://github.com/Z-Anatomy/Models-of-human-anatomy. The source
  catalog declares revision 9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a and
  binds it to Z-Anatomy.zip (SHA-256
  e029688545627bd0214b269e1063143abb580aad72b2c2445d6d8a9a0d9da736) and
  Z-Anatomy/Startup.blend (SHA-256
  9f08a17ea0115fed80b2a73ecdf0a1bc2ab2f6956f37c593ce23d513ea35afcd). The
  revision is catalog-declared and archive-bound; it is not written into the
  raw export JSON as a Git checkout revision.
- License: the pinned revision's [License.txt](https://github.com/Z-Anatomy/Models-of-human-anatomy/blob/9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a/License.txt)
  identifies Z-Anatomy material as CC BY-SA 4.0 and requires attribution and
  ShareAlike for derivatives. It separately credits BodyParts3D material
  included in Z-Anatomy under CC BY-SA 2.1 Japan. That inherited credit is not
  the same as the direct BodyParts3D 4.0 archives listed above.
- Required Z-Anatomy attribution from that pinned license:
  Z-Anatomy - The libre 3D atlas of anatomy - CC-BY-SA 4.0.
- Required inherited BodyParts3D attribution from the same pinned license:
  BodyParts3D - The Database Center for Life Science - CC-BY-SA 2.1 Japan.
- Resting-scene material: the current anatomy receipt identifies stable IDs
  305–309 as five lung-lobe source surfaces, 310 as the derived pleura proxy,
  and 311 as diaphragm source geometry. IDs 14–21 are inferred exterior display
  patches whose names/FMA crosswalks use Z-Anatomy segment guides; the rendered
  geometry is one closed aggregate liver shell, not eight source segment
  meshes, internal partitions, or segment volumes. The selected Z-derived
  scene content does not include the separately credited cranial-nerve,
  inner-ear, or kidney reference components; the latter two carry additional
  non-commercial terms in the upstream license.
- The pleura is an external lung-union proxy, not a parietal pleural layer,
  interlobar lining, fluid domain, or pleural mechanics model. Lung meshes are
  registered atlas envelopes, not parenchymal tissue, measured lung volumes,
  or clinical registration.
- Tracked calf derivatives remain the right lateral/medial gastrocnemius,
  soleus, calcaneal tendon, and matching right calcaneus. The calcaneus is a
  rigid visual overlay bound to the existing MyoSim calcn_r body; it is not a
  replacement force or mechanics asset. The original and smoothed calf
  inspections are under
  Docs/media/myosim-native-zanatomy-matched-calcaneus-2048/ and
  Docs/media/myosim-native-zanatomy-smooth-insertion-2048/.
- The current resting-scene payload retains a historical stable-ID-22 row whose
  source identity is BodyParts3D member FJ2824. The retained source map and
  neutral-pose rebase record trace its unchanged face topology from body 7 to
  torso body 20; the current receipt intentionally retires ID 22 as an alias to
  liver display patch 21 and excludes it from the active source-ID map. This
  alias does not define a second liver segment or volume. See the companion
  scene attribution record for the exact payload and receipt hashes.

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
- License: [Apache License 2.0](https://github.com/MyoHub/myo_sim/blob/33c89c2bde282553dde3f526768eb3bdcfaa7649/LICENSE). The source archive and all generated local
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
