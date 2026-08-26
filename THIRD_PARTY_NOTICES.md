# Third-party source notices

No third-party source data is committed to this repository. The importer records
the following sources in each generated local manifest.

## BodyParts3D 4.0

- Upstream: <https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html>
- License: [CC BY 4.0](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html)
- Required attribution: `BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International`.
- Imported material: 4.0 OBJ meshes, FMA identifiers, English labels, and both
  `is-a` and `part-of` hierarchy relationships.

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

## Numi Lab boundary

The resulting manifest is a source-faithful import artifact, not a validated
medical model, real-time Numi simulation, material calibration, or clinical
claim.
